// DTN engine: Dominance-Tree Network inference engine of the RV32I-Alpha SoC.
// Engineer: Badreddine BENFETTOUMA (University of Bourgogne Europe)
// Normative specification: docs/dtn/SPEC.md (v1). Outputs, RESULT, MAXACC and WORDS are exactly
// those of SPEC 5; WORDS = number of cycles with w_en = 1.
//
// ---------------------------------------------------------------------------------------------
// Arithmetic (all lane sums are unsigned, one adder tree of LANES bytes)
//   The root accumulator racc holds the certification margin, not acc:
//       racc = acc + (2^p - 1)*Spos - T0 - 1        (RELU; T0 term replaced by 0 for ARGMAX)
//   so that SPEC "cert" (acc + (2^p-1)*Spos <= T0) is simply racc < 0 (its sign bit).
//   - bias word        : racc = b,                 cacc = -T0 - 1 (= ~T0; 0 for ARGMAX)
//   - sign word, grp g : racc += 128*Spos_g,       cacc -= Spos_g    Spos_g = sum x[i], sw[i]=0
//   - CORR (no read)   : racc += cacc   -> b + 127*Spos - T0 - 1, the bound after the sign plane
//   - magnitude plane p: racc -= 2^p * sum x[i] over the lanes with mw[i] == sw[i]
//     (exact: acc += 2^p*L_p and the bound loses 2^p*Spos, and L_p - Spos = -sum x[mw==sw])
//   After plane 0 the bound term is 0: racc = acc - T0 - 1 exactly (ARGMAX: racc = acc).
//   The child accumulator cacc holds a - T0 - 1:
//   - db word          : cacc = racc - db;   gap plane p: cacc -= 2^p * sum x[i] over dw[i] = 1
//   so "ccert" (a <= T0) is cacc < 0. Every certification test is one sign bit.
//   34-bit accumulators: exact for any int32 T0 and any acc that fits in int32 (SPEC 2).
//   The requantiser gets acc back as u + T0 + 1 (dtn_requant.sv, sequential, ~17 cycles).
//
// Pipeline (one weight word per cycle in steady state)
//   A: w_en / w_addr registers (the word is read)      op descriptor a_*
//   B: w_rdata valid: group read of the input buffer, lane enables, adder tree -> breg
//   C: racc / cacc update; the sign of the adder output is the certification decision
//   D: end of an item: requantisation job (RELU, if active) or argmax update (ARGMAX)
//   A decision (RELU only) stops the issue until the deciding word is in C: 2 empty bus
//   cycles per plane boundary. Only real decisions wait: none in ARGMAX, none after the last
//   gap plane of a child, none after magnitude plane 0 when C = 0.
//
// Cycles of a layer (G' = number of groups with nz = 1, all counts after PREP)
//   PREP (nz[g], clear of the output bytes < R*(1+C))  : G_MAX + 4 cycles, end: 1 cycle
//   root (RELU), stopped after the sign plane          : G' + 4
//   root (RELU), stopped after magnitude plane p       : 1 + G' + 3 + (7-p)*(G' + 2)
//   root (RELU), active, C > 0                         : 8*G' + 18  (C = 0: 8*G' + 16)
//   root (ARGMAX)                                      : 8*G' + 2
//   child, stopped after db                            : 3
//   child, stopped after gap plane p                   : 3 + (NB-p)*(G' + 2)
//   child, active                                      : NB*G' + 2*NB + 1
//   (G' = 0: root = bias + CORR [+2 if it must wait], child = db only.)
//   plus stalls when an active item ends while the requantiser is still busy with the
//   previous one (it accepts a job every max(SH, 15) + 2 cycles).
//
// Activation buffers: two register arrays of LANES*G_MAX bytes. One LANES-byte group read port
// (shared): the engine (stage B) while BUSY, the bus otherwise. Consequence: bus reads of BUF0/
// BUF1 while BUSY return undefined data (writes are ignored while BUSY anyway, SPEC 6). Bus
// addresses beyond LANES*G_MAX bytes in a buffer window: writes ignored, reads return 0.
// Output bytes < R*(1+C) are cleared in PREP (RELU), so certified outputs need no write; the
// requantiser writes the active ones.
// ---------------------------------------------------------------------------------------------

module dtn_engine #(
    parameter int G_MAX = 4,    // activation groups (N_MAX = LANES*G_MAX)
    parameter int AW    = 12,   // weight memory word address width
    parameter int LANES = 32    // lanes per weight word (32 in silicon; smaller only for formal)
) (
    input  logic             clk,
    input  logic             rst,       // active high, asynchronous assert
    // slave port from the SoC (same contract as the core's dmem port, always granted):
    input  logic             s_en,
    input  logic [3:0]       s_we,
    // s_addr[1:0] are not used: every access is word aligned, s_we selects the bytes
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [11:0]      s_addr,    // byte offset in the window
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [31:0]      s_wdata,
    output logic [31:0]      s_rdata,   // valid in the cycle after the request
    // weight memory read port (synchronous, data valid in the cycle after w_en, not held after)
    output logic             w_en,
    output logic [AW-1:0]    w_addr,
    input  logic [LANES-1:0] w_rdata,
    output logic             done       // = STATUS.DONE
);
    // ----------------------------------------------------------------------------- sizes
    localparam int NBUF = LANES * G_MAX;                   // bytes per activation buffer
    localparam int LW   = $clog2(LANES);                    // lane index bits
    localparam int GW   = (G_MAX > 1) ? $clog2(G_MAX) : 1;  // group index bits
    localparam int NW   = $clog2(NBUF + 1);                 // N register bits (1 .. NBUF)
    localparam int GCW  = NW - LW + 1;                      // bits of G = ceil(N / LANES)
    localparam int OW   = (NBUF > 1) ? $clog2(NBUF) : 1;    // output byte index bits
    localparam int SW   = 8 + LW;                           // lane sum bits
    localparam int ACCW = 34;                               // accumulator bits (see header)
    localparam int BW   = (LANES > SW) ? LANES : SW;        // stage B -> C data register
    localparam int DW   = GCW + 1;                          // address step bits
    localparam int LWPG = LW - 2;                           // log2(bus words per group)
    localparam int WW   = (LWPG > 0) ? LWPG : 1;            // word-in-group index bits
    localparam int HW   = 13 - LW;                          // lane_mask: count >> LW

    // op descriptors carried by the pipeline
    localparam logic [2:0] OP_REC  = 3'd0;   // bias (root) or db (child) word
    localparam logic [2:0] OP_SIGN = 3'd1;   // root sign plane word
    localparam logic [2:0] OP_MAG  = 3'd2;   // root magnitude plane word
    localparam logic [2:0] OP_GAP  = 3'd3;   // child gap plane word
    localparam logic [2:0] OP_CORR = 3'd4;   // no read: racc += cacc (end of the sign plane)
    localparam logic [2:0] OP_PREP = 3'd5;   // no read: nz[g] of input group g

    // issue position (what stage A issues next)
    localparam logic [1:0] PK_REC   = 2'd0;  // record word (bias / db) of the current item
    localparam logic [1:0] PK_PLANE = 2'd1;  // word (pl, grp) of the current item
    localparam logic [1:0] PK_CORR  = 2'd2;  // root CORR op
    localparam logic [1:0] PK_NEXT  = 2'd3;  // current item finished (all planes read)

    // control states
    localparam logic [2:0] S_IDLE  = 3'd0;
    localparam logic [2:0] S_PREP  = 3'd1;   // nz[] and output clear, one group per cycle
    localparam logic [2:0] S_PWAIT = 3'd2;   // last PREP op reaches stage C
    localparam logic [2:0] S_RUN   = 3'd3;   // issue
    localparam logic [2:0] S_WAIT  = 3'd4;   // wait for a certification decision in stage C
    localparam logic [2:0] S_FIN   = 3'd5;   // drain the pipeline and the requantiser

    // lane i of group g holds input LANES*g + i: valid when that index is < cnt
    function automatic logic [LANES-1:0] lane_mask(input logic [12:0] cnt, input logic [GW-1:0] g);
        for (int i = 0; i < LANES; i++)
            lane_mask[i] = (cnt[12:LW] > HW'(g)) || ((cnt[12:LW] == HW'(g)) && (cnt[LW-1:0] > LW'(i)));
    endfunction

    // byte-enable merge of a register write
    function automatic logic [31:0] merge(input logic [31:0] old, input logic [31:0] nw,
                                          input logic [3:0] be);
        for (int k = 0; k < 4; k++)
            merge[8*k +: 8] = be[k] ? nw[8*k +: 8] : old[8*k +: 8];
    endfunction

    // ----------------------------------------------------------------------------- state
    logic [2:0]          st;
    logic                busy, done_q;
    logic                kind_q, inb_q, relu;            // latched at START

    // MMIO registers
    logic [AW-1:0]       wbase_q;
    logic [NW-1:0]       n_q;
    logic [7:0]          r_q;
    logic [3:0]          c_q, nb_q;
    logic [14:0]         m_q;
    logic [4:0]          sh_q;
    logic [31:0]         t0_q;
    logic [7:0]          result_q;
    logic [31:0]         maxacc_q;
    logic [23:0]         words_q, cycles_q;              // bound: R*S_TREE words, << 2^24

    // layer constants
    logic [GCW-1:0]      g_q;                            // G = ceil(N / LANES)
    logic [AW-1:0]       s_root, s_child_q, s_tree_q;    // SPEC 3 record sizes
    logic [11:0]         nout_q;                         // R*(1+C)

    // activation buffers, sign plane of the current root, group pattern
    logic [8*NBUF-1:0]   buf0_q, buf1_q;
    logic [LANES*G_MAX-1:0] sw_q;
    logic [G_MAX-1:0]    nz;

    // issue control (stage A)
    logic [GW-1:0]       pg;                             // PREP group
    logic [1:0]          pk;
    logic                it_child;                       // current item: 0 root, 1 child
    logic [2:0]          pl;                             // plane (root: 7 = sign, 6..0)
    logic [GW-1:0]       grp;                            // group of the next plane word
    logic [7:0]          ridx;
    logic [3:0]          cidx;
    logic [OW-1:0]       oidx, oroot, fin_oidx;          // output index: item, root, job
    logic [DW-1:0]       dnext;                          // address step to the next plane word
    logic [AW-1:0]       rbase, cbase;                   // next root record, next child record
    logic                rb_pend;

    // pipeline
    logic                a_v, b_v, c_v, d_v;
    logic [2:0]          a_op, b_op, c_op;
    logic                a_ch, b_ch, c_ch, d_ch;
    logic [2:0]          a_sh, b_sh, c_sh;
    logic [GW-1:0]       a_g, b_g, c_g;
    logic                a_dec, b_dec, c_dec;
    logic                a_fin, b_fin, c_fin;
    logic [BW-1:0]       breg;
    logic [ACCW-1:0]     racc, cacc;

    // requantiser
    logic                q_start, q_idle, q_wr;
    logic [7:0]          q_val;

    assign busy = (st != S_IDLE);
    assign relu = ~kind_q;
    assign done = done_q;

    // ============================================================================= bus slave
    logic wr_req, reg_wr, start;
    assign wr_req = s_en & (s_we != 4'd0) & ~busy;         // writes are ignored while BUSY
    assign reg_wr = wr_req & (s_addr[11:10] == 2'b00);
    assign start  = reg_wr & (s_addr[9:2] == 8'h00) & s_we[0] & s_wdata[0];

    always_ff @(posedge clk)
        if (reg_wr)
            case (s_addr[9:2])
                8'h02:   wbase_q <= AW'(merge(32'(wbase_q), s_wdata, s_we));
                8'h03:   n_q     <= NW'(merge(32'(n_q), s_wdata, s_we));
                8'h04:   r_q     <= 8'(merge(32'(r_q), s_wdata, s_we));
                8'h05:   c_q     <= 4'(merge(32'(c_q), s_wdata, s_we));
                8'h06:   nb_q    <= 4'(merge(32'(nb_q), s_wdata, s_we));
                8'h07:   m_q     <= 15'(merge(32'(m_q), s_wdata, s_we));
                8'h08:   sh_q    <= 5'(merge(32'(sh_q), s_wdata, s_we));
                8'h09:   t0_q    <= merge(t0_q, s_wdata, s_we);
                default: ;
            endcase

    // START: latch the layer kind, the buffers and the record sizes
    logic [GCW-1:0] g_new;
    logic [GCW+3:0] nbg;
    assign g_new  = GCW'(n_q[NW-1:LW]) + GCW'(n_q[LW-1:0] != '0);    // ceil(N / LANES)
    assign nbg    = (GCW+4)'(nb_q) * (GCW+4)'(g_new);
    assign s_root = AW'({g_q, 3'b001});                    // 1 + 8*G

    always_ff @(posedge clk) begin
        if (start) begin
            kind_q    <= s_wdata[1];
            inb_q     <= s_wdata[2];
            g_q       <= g_new;
            s_child_q <= AW'(nbg) + AW'(1);                  // 1 + NB*G
            nout_q    <= 12'(r_q) * (12'(c_q) + 12'd1);
        end
        if (st == S_PREP)
            s_tree_q  <= s_root + AW'(c_q) * s_child_q;      // S_ROOT + C*S_CHILD
    end

    // read: the address is registered, the data is selected in the next cycle
    logic [9:0]        rd_a;                               // word address of the last request
    logic [GW-1:0]     rd_grp;
    logic [WW-1:0]     rd_wig;
    logic              rd_in;                              // inside the buffer
    logic [8*LANES-1:0] gx;                                // group read port
    logic [31:0]       bufword;

    always_ff @(posedge clk)
        if (s_en) rd_a <= s_addr[11:2];

    assign rd_grp = rd_a[LWPG +: GW];
    assign rd_in  = (rd_a[7:0] < 8'(NBUF / 4));
    if (LWPG > 0) begin : g_wig
        assign rd_wig = rd_a[WW-1:0];
    end else begin : g_nowig
        assign rd_wig = 1'b0;
    end

    always_comb begin
        bufword = 32'd0;
        for (int k = 0; k < LANES / 4; k++)
            if (rd_wig == WW'(k)) bufword = gx[32*k +: 32];
    end

    always_comb begin
        s_rdata = 32'd0;
        case (rd_a[9:8])
            2'b01, 2'b10: if (rd_in) s_rdata = bufword;
            2'b00:
                case (rd_a[7:0])
                    8'h00:   s_rdata = {29'd0, inb_q, kind_q, 1'b0};
                    8'h01:   s_rdata = {30'd0, done_q, busy};
                    8'h02:   s_rdata = 32'(wbase_q);
                    8'h03:   s_rdata = 32'(n_q);
                    8'h04:   s_rdata = 32'(r_q);
                    8'h05:   s_rdata = 32'(c_q);
                    8'h06:   s_rdata = 32'(nb_q);
                    8'h07:   s_rdata = 32'(m_q);
                    8'h08:   s_rdata = 32'(sh_q);
                    8'h09:   s_rdata = t0_q;
                    8'h0A:   s_rdata = 32'(result_q);
                    8'h0B:   s_rdata = maxacc_q;
                    8'h0C:   s_rdata = 32'(words_q);
                    8'h0D:   s_rdata = 32'(cycles_q);
                    default: s_rdata = 32'd0;
                endcase
            default: ;
        endcase
    end

    always_ff @(posedge clk)
        if (start) begin
            words_q  <= 24'd0;
            cycles_q <= 24'd0;
        end else begin
            if (w_en) words_q  <= words_q + 24'd1;
            if (busy) cycles_q <= cycles_q + 24'd1;
        end

    // ============================================================================= buffers
    // group read: input group b_g while BUSY, the group of the bus address otherwise
    logic [GW-1:0] gsel;
    logic          bsel;
    assign gsel = busy ? b_g : rd_grp;
    assign bsel = busy ? inb_q : rd_a[9];
    always_comb begin
        gx = '0;
        for (int g = 0; g < G_MAX; g++)
            gx = gx | ({8*LANES{gsel == GW'(g)}} &
                       (bsel ? buf1_q[8*LANES*g +: 8*LANES] : buf0_q[8*LANES*g +: 8*LANES]));
    end

    // writes: the bus (idle), the PREP clear and the requantiser (busy, output buffer only)
    logic             clr;
    logic [LANES-1:0] clr_mask;
    logic [7:0]       eng_wd;
    logic [NBUF-1:0]  we0, we1;
    assign clr      = (st == S_PREP) & relu;
    assign clr_mask = lane_mask(13'(nout_q), pg);
    assign eng_wd   = clr ? 8'd0 : q_val;

    always_comb
        for (int j = 0; j < NBUF; j++) begin
            we0[j] = wr_req & (s_addr[11:10] == 2'b01) & (s_addr[9:2] == 8'(j / 4)) & s_we[j % 4];
            we1[j] = wr_req & (s_addr[11:10] == 2'b10) & (s_addr[9:2] == 8'(j / 4)) & s_we[j % 4];
            if ((clr & (pg == GW'(j / LANES)) & clr_mask[j % LANES]) | (q_wr & (fin_oidx == OW'(j)))) begin
                if (inb_q) we0[j] = 1'b1;
                else       we1[j] = 1'b1;
            end
        end

    always_ff @(posedge clk)
        for (int j = 0; j < NBUF; j++) begin
            if (we0[j]) buf0_q[8*j +: 8] <= busy ? eng_wd : s_wdata[8*(j % 4) +: 8];
            if (we1[j]) buf1_q[8*j +: 8] <= busy ? eng_wd : s_wdata[8*(j % 4) +: 8];
        end

    // ============================================================================= stage B
    logic [LANES-1:0] sw_g, ven, len;
    logic [SW-1:0]    tsum;

    always_comb begin
        sw_g = '0;
        for (int g = 0; g < G_MAX; g++)
            if (b_g == GW'(g)) sw_g = sw_q[LANES*g +: LANES];
    end
    assign ven = lane_mask(13'(n_q), b_g);                 // x[j] = 0 for j >= N

    always_comb
        case (b_op)
            OP_PREP: len = ven;                            // nz[g]
            OP_SIGN: len = ven & ~w_rdata;                 // Spos_g
            OP_MAG:  len = ven & ~(w_rdata ^ sw_g);        // lanes with mw == sw
            OP_GAP:  len = ven & w_rdata;                  // D.x
            default: len = '0;
        endcase

    dtn_lane_sum #(.LANES(LANES)) u_sum (.x(gx), .en(len), .sum(tsum));

    always_ff @(posedge clk)
        if (b_v) breg <= (b_op == OP_REC) ? BW'(w_rdata) : BW'(tsum);

    always_ff @(posedge clk)
        for (int g = 0; g < G_MAX; g++)
            if (b_v && b_op == OP_SIGN && b_g == GW'(g)) sw_q[LANES*g +: LANES] <= w_rdata;

    // ============================================================================= stage C
    logic [SW-1:0]   lsum;
    logic [SW+6:0]   shl;
    logic [ACCW-1:0] bias_x, db_x, rb, rsum, ca, cb, csum;
    logic            c_child, cert;

    assign lsum    = breg[SW-1:0];
    assign shl     = (SW+7)'(lsum) << c_sh;
    assign bias_x  = {{(ACCW-LANES){breg[LANES-1]}}, breg[LANES-1:0]};   // b: signed
    assign db_x    = {{(ACCW-LANES){1'b0}}, breg[LANES-1:0]};            // db: unsigned
    assign c_child = (c_op == OP_GAP) | ((c_op == OP_REC) & c_ch);

    always_comb
        case (c_op)
            OP_SIGN: rb = ACCW'(shl);                      // + 128*Spos_g
            OP_MAG:  rb = ~(ACCW'(shl));                   // - 2^p * sum (with the +1 below)
            OP_CORR: rb = cacc;
            default: rb = '0;
        endcase
    assign rsum = racc + rb + ACCW'(c_op == OP_MAG);

    assign ca = (c_op == OP_REC) ? racc : cacc;
    always_comb
        case (c_op)
            OP_REC:  cb = db_x;
            OP_SIGN: cb = ACCW'(lsum);
            default: cb = ACCW'(shl);
        endcase
    assign csum = ca - cb;
    assign cert = c_child ? csum[ACCW-1] : rsum[ACCW-1];   // decision of a c_dec op

    always_ff @(posedge clk)
        if (c_v)
            case (c_op)
                OP_REC:
                    if (c_ch) cacc <= csum;
                    else begin
                        racc <= bias_x;
                        cacc <= relu ? {{(ACCW-32){~t0_q[31]}}, ~t0_q} : '0;
                    end
                OP_SIGN: begin
                    racc <= rsum;
                    cacc <= csum;
                end
                OP_MAG, OP_CORR: racc <= rsum;
                OP_GAP:  cacc <= csum;
                default: ;
            endcase

    always_ff @(posedge clk)
        for (int g = 0; g < G_MAX; g++)
            if (c_v && c_op == OP_PREP && c_g == GW'(g)) nz[g] <= (lsum != '0);

    // ============================================================================= stage D
    logic [ACCW-1:0] dacc;
    logic [7:0]      am_r;                                 // ARGMAX row of the stage D op
    assign dacc    = d_ch ? cacc : racc;
    assign q_start = d_v & relu & ~dacc[ACCW-1];           // active: requantise

    always_ff @(posedge clk)
        if (start) am_r <= 8'd0;
        else if (d_v && !relu) begin
            am_r <= am_r + 8'd1;
            if (am_r == 8'd0 || $signed(racc[31:0]) > $signed(maxacc_q)) begin   // first max wins
                maxacc_q <= racc[31:0];
                result_q <= am_r;
            end
        end

    dtn_requant u_rq (
        .clk(clk), .rst(rst), .start(q_start), .u(dacc[31:0]), .t0(t0_q), .m(m_q), .sh(sh_q),
        .idle(q_idle), .wr(q_wr), .q(q_val)
    );

    // ============================================================================= stage A
    // group pattern of the layer
    logic          any_nz;
    logic [GW-1:0] g_first, g_last, g_nxt;
    logic [DW-1:0] d_rec, d_wrap, d_step;
    always_comb begin
        g_first = '0;
        g_last  = '0;
        g_nxt   = '0;
        for (int g = G_MAX - 1; g >= 0; g--) if (nz[g]) g_first = GW'(g);
        for (int g = 0; g < G_MAX; g++)      if (nz[g]) g_last  = GW'(g);
        for (int g = G_MAX - 1; g >= 0; g--) if (nz[g] && GW'(g) > grp) g_nxt = GW'(g);
    end
    assign any_nz = (nz != '0);
    assign d_rec  = DW'(g_first) + DW'(1);                     // record -> first plane word
    assign d_wrap = DW'(g_q) - DW'(g_last) + DW'(g_first);     // last group -> next plane
    assign d_step = DW'(g_nxt) - DW'(grp);                     // next group, same plane

    // a job may be issued only if the requantiser is free when it reaches stage D
    logic core_free;
    assign core_free = (q_idle | q_wr) & ~(a_v & a_fin) & ~(b_v & b_fin) & ~(c_v & c_fin) & ~d_v;

    // effective position: the registers, or the next item once the current one is finished
    // (PK_NEXT) or certified (decision with cert = 1)
    logic          e_ok, e_skip, e_end, e_ch, e_fin, e_go;
    logic [1:0]    e_pk;
    logic [7:0]    e_ridx;
    logic [3:0]    e_cidx;
    logic [OW-1:0] e_oidx, e_oroot;
    always_comb begin
        e_ok    = (st == S_RUN) | ((st == S_WAIT) & c_v & c_dec);
        e_skip  = (st == S_WAIT) & cert;
        e_end   = 1'b0;
        e_pk    = pk;
        e_ch    = it_child;
        e_ridx  = ridx;
        e_cidx  = cidx;
        e_oidx  = oidx;
        e_oroot = oroot;
        if (e_skip || pk == PK_NEXT) begin
            e_pk = PK_REC;
            if (!it_child && !e_skip && relu && c_q != 4'd0) begin         // active root: child 0
                e_ch   = 1'b1;
                e_cidx = 4'd0;
                e_oidx = oroot + OW'(1);
            end else if (it_child && cidx != c_q - 4'd1) begin             // next child
                e_cidx = cidx + 4'd1;
                e_oidx = oidx + OW'(1);
            end else if (ridx == r_q - 8'd1) begin                         // layer finished
                e_end  = 1'b1;
            end else begin                                                  // next root
                e_ch    = 1'b0;
                e_ridx  = ridx + 8'd1;
                e_oroot = oroot + OW'(c_q) + OW'(1);
                e_oidx  = e_oroot;
            end
        end
        // the last op of an item that reads all its planes ends with a job (RELU, if active)
        e_fin = ((e_pk == PK_REC) & e_ch & (~any_nz | (nb_q == 4'd0)))
              | ((e_pk == PK_PLANE) & (pl == 3'd0) & (grp == g_last))
              | ((e_pk == PK_CORR) & ~any_nz);
        e_go  = e_ok & ~e_end & (~(relu & e_fin) | core_free);
    end

    // next-state of stage A
    logic [2:0]    st_d;
    logic [GW-1:0] pg_d, grp_d, a_g_d;
    logic [1:0]    pk_d;
    logic          ch_d, rb_set, cb_child;
    logic [2:0]    pl_d, a_op_d, a_sh_d;
    logic [7:0]    ridx_d;
    logic [3:0]    cidx_d;
    logic [OW-1:0] oidx_d, oroot_d, fin_oidx_d;
    logic [DW-1:0] dnext_d;
    logic          a_v_d, a_ch_d, a_dec_d, a_fin_d, w_en_d;
    logic [AW-1:0] w_addr_d;

    always_comb begin
        st_d       = st;
        pg_d       = pg;
        pk_d       = pk;
        ch_d       = it_child;
        pl_d       = pl;
        grp_d      = grp;
        ridx_d     = ridx;
        cidx_d     = cidx;
        oidx_d     = oidx;
        oroot_d    = oroot;
        fin_oidx_d = fin_oidx;
        dnext_d    = dnext;
        a_v_d      = 1'b0;
        a_op_d     = OP_REC;
        a_ch_d     = 1'b0;
        a_sh_d     = pl;
        a_g_d      = grp;
        a_dec_d    = 1'b0;
        a_fin_d    = 1'b0;
        w_en_d     = 1'b0;
        w_addr_d   = w_addr;
        rb_set     = 1'b0;      // root record issued: cbase = rbase + S_ROOT, then rbase += S_TREE
        cb_child   = 1'b0;      // child record issued: cbase += S_CHILD

        case (st)
            S_IDLE:
                if (start) begin
                    st_d    = S_PREP;
                    pg_d    = '0;
                    pk_d    = PK_REC;
                    ch_d    = 1'b0;
                    ridx_d  = 8'd0;
                    cidx_d  = 4'd0;
                    oidx_d  = '0;
                    oroot_d = '0;
                end
            S_PREP: begin
                a_v_d  = 1'b1;
                a_op_d = OP_PREP;
                a_g_d  = pg;
                pg_d   = pg + GW'(1);
                if (pg == GW'(G_MAX - 1)) st_d = S_PWAIT;
            end
            S_PWAIT:
                if (!a_v && !b_v) st_d = (r_q == 8'd0) ? S_FIN : S_RUN;   // nz ready next cycle
            S_FIN:
                if (!a_v && !b_v && !c_v && !d_v && q_idle) st_d = S_IDLE;
            default:                                                       // S_RUN, S_WAIT
                if (e_ok) begin
                    pk_d    = e_pk;
                    ch_d    = e_ch;
                    ridx_d  = e_ridx;
                    cidx_d  = e_cidx;
                    oidx_d  = e_oidx;
                    oroot_d = e_oroot;
                    st_d    = e_end ? S_FIN : S_RUN;
                    if (e_go) begin
                        a_v_d   = 1'b1;
                        a_fin_d = e_fin;
                        if (e_fin) fin_oidx_d = e_oidx;
                        case (e_pk)
                            PK_REC: begin
                                w_en_d   = 1'b1;
                                w_addr_d = e_ch ? cbase : rbase;
                                a_op_d   = OP_REC;
                                a_ch_d   = e_ch;
                                grp_d    = g_first;
                                dnext_d  = d_rec;
                                if (!e_ch) begin
                                    rb_set = 1'b1;
                                    if (any_nz) begin
                                        pk_d = PK_PLANE;
                                        pl_d = 3'd7;                               // sign plane
                                    end else
                                        pk_d = PK_CORR;
                                end else begin
                                    cb_child = 1'b1;
                                    if (any_nz && nb_q != 4'd0) begin
                                        pk_d    = PK_PLANE;
                                        pl_d    = 3'(nb_q - 4'd1);
                                        a_dec_d = 1'b1;                            // a <= T0 ?
                                        st_d    = S_WAIT;
                                    end else
                                        pk_d = PK_NEXT;
                                end
                            end
                            PK_PLANE: begin
                                w_en_d   = 1'b1;
                                w_addr_d = w_addr + AW'(dnext);
                                a_op_d   = it_child ? OP_GAP : ((pl == 3'd7) ? OP_SIGN : OP_MAG);
                                if (grp != g_last) begin
                                    grp_d   = g_nxt;
                                    dnext_d = d_step;
                                end else begin                                     // plane end
                                    grp_d   = g_first;
                                    dnext_d = d_wrap;
                                    if (!it_child && pl == 3'd7)
                                        pk_d = PK_CORR;
                                    else if (pl != 3'd0) begin
                                        pl_d    = pl - 3'd1;
                                        a_dec_d = relu;
                                        if (relu) st_d = S_WAIT;
                                    end else begin
                                        pk_d = PK_NEXT;
                                        if (relu && !it_child && c_q != 4'd0) begin  // children?
                                            a_dec_d = 1'b1;
                                            st_d    = S_WAIT;
                                        end
                                    end
                                end
                            end
                            PK_CORR: begin
                                a_op_d = OP_CORR;
                                if (any_nz) begin
                                    pk_d = PK_PLANE;
                                    pl_d = 3'd6;               // grp and dnext already point there
                                end else
                                    pk_d = PK_NEXT;
                                if (relu && (any_nz || c_q != 4'd0)) begin
                                    a_dec_d = 1'b1;
                                    st_d    = S_WAIT;
                                end
                            end
                            default: ;                         // PK_NEXT is resolved above
                        endcase
                    end
                end
        endcase
    end

    // record bases (one AW-bit adder): rbase always holds the base of the NEXT root once the
    // current root record is issued, cbase the base of the next child
    logic [AW-1:0] base_sum;
    assign base_sum = rb_pend  ? rbase + s_tree_q :
                      cb_child ? cbase + s_child_q : rbase + s_root;
    always_ff @(posedge clk) begin
        if (start)        rbase <= wbase_q;
        else if (rb_pend) rbase <= base_sum;
        if (rb_set | cb_child) cbase <= base_sum;
    end

    // control registers (asynchronous reset)
    always_ff @(posedge clk or posedge rst)
        if (rst) begin
            st      <= S_IDLE;
            done_q  <= 1'b0;
            a_v     <= 1'b0;
            b_v     <= 1'b0;
            c_v     <= 1'b0;
            d_v     <= 1'b0;
            w_en    <= 1'b0;
            rb_pend <= 1'b0;
        end else begin
            st      <= st_d;
            if (start)                                  done_q <= 1'b0;
            else if (st == S_FIN && st_d == S_IDLE)     done_q <= 1'b1;
            a_v     <= a_v_d;
            b_v     <= a_v;
            c_v     <= b_v;
            d_v     <= c_v & c_fin;
            w_en    <= w_en_d;
            rb_pend <= rb_set;
        end

    // data registers of stage A and of the pipeline (qualified by the valid bits)
    always_ff @(posedge clk) begin
        pg       <= pg_d;
        pk       <= pk_d;
        it_child <= ch_d;
        pl       <= pl_d;
        grp      <= grp_d;
        ridx     <= ridx_d;
        cidx     <= cidx_d;
        oidx     <= oidx_d;
        oroot    <= oroot_d;
        fin_oidx <= fin_oidx_d;
        dnext    <= dnext_d;
        w_addr   <= w_addr_d;
        a_op  <= a_op_d;  a_ch  <= a_ch_d;  a_sh  <= a_sh_d;  a_g  <= a_g_d;
        a_dec <= a_dec_d; a_fin <= a_fin_d;
        b_op  <= a_op;    b_ch  <= a_ch;    b_sh  <= a_sh;    b_g  <= a_g;
        b_dec <= a_dec;   b_fin <= a_fin;
        c_op  <= b_op;    c_ch  <= b_ch;    c_sh  <= b_sh;    c_g  <= b_g;
        c_dec <= b_dec;   c_fin <= b_fin;
        d_ch  <= c_child;
    end
endmodule
