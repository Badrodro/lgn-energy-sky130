// DTN engine: sequential requantisation.
// Engineer: Badreddine BENFETTOUMA (University of Bourgogne Europe)
//
// q = clamp((acc * M + 2^(SH-1)) >> SH, 0, 255)            (SPEC 2, '>>' arithmetic)
//
// The engine hands over the accumulator as a margin u = acc - T0 - 1 (low 32 bits). One job:
//   cycle 0 (start, idle)  acc = u + T0 + 1 (mod 2^32, exact because acc fits in int32)
//   MUL, one bit of M per cycle, LSB first, max(SH, bits of M) cycles (15 for SH <= 15):
//     step k < SH  ("right"): P = (P + M[k]*acc) >> 1, the bit shifted out is kept in rbit.
//                  After SH steps P = floor(Y_SH / 2^SH) and rbit = bit SH-1 of Y = acc*M.
//     step k >= SH ("left") : P = P + M[k]*(acc << (k-SH)); the multiplicand shifts left.
//                  Every term is >= 0 here, so an overflow only needs a sticky 'sat' flag.
//   WR: floor((Y + 2^(SH-1)) / 2^SH) = floor(Y / 2^SH) + Y[SH-1] = P + rbit, clamped.
// A negative acc gives q = 0 directly (M > 0, so acc*M <= 0); acc = 0 gives P = 0.
// One 33-bit adder, no barrel shifter, no wide product register.

module dtn_requant (
    input  logic        clk,
    input  logic        rst,       // active high, asynchronous assert
    input  logic        start,     // new job (taken only when idle)
    input  logic [31:0] u,         // acc - T0 - 1
    input  logic [31:0] t0,
    input  logic [14:0] m,
    input  logic [4:0]  sh,
    output logic        idle,      // ready for a job
    output logic        wr,        // q is valid: write it into the output buffer this cycle
    output logic [7:0]  q
);
    localparam logic [1:0] Q_IDLE = 2'd0;
    localparam logic [1:0] Q_MUL  = 2'd1;
    localparam logic [1:0] Q_WR   = 2'd2;

    logic [1:0]  state;
    logic [31:0] p;          // partial result
    logic [31:0] mc;         // multiplicand: acc, shifted left in the left steps
    logic [14:0] mr;         // remaining multiplier bits (LSB = current bit)
    logic [4:0]  shc;        // right steps still to do
    logic        rbit;       // rounding bit Y[SH-1]
    logic        sat;        // result >= 2^32 (left steps only)
    logic        lost;       // a 1 was shifted out of mc
    logic        neg;        // acc < 0

    // one adder: acc = u + T0 + 1 when idle, P + M[k]*mc otherwise
    logic        load;
    logic [31:0] add_a, add_b;
    logic [32:0] add_s;
    assign load  = (state == Q_IDLE);
    assign add_a = load ? u : p;
    assign add_b = load ? t0 : (mr[0] ? mc : 32'd0);
    assign add_s = {1'b0, add_a} + {1'b0, add_b} + {32'd0, load};

    logic right, last;
    assign right = (shc != 5'd0);
    assign last  = (shc <= 5'd1) & (mr[14:1] == 14'd0);   // nothing left after this step

    always_ff @(posedge clk or posedge rst)
        if (rst) state <= Q_IDLE;
        else case (state)
            Q_IDLE:  if (start) state <= Q_MUL;
            Q_MUL:   if (neg || last) state <= Q_WR;
            default: state <= Q_IDLE;
        endcase

    always_ff @(posedge clk) begin
        if (load) begin
            if (start) begin
                mc   <= add_s[31:0];
                neg  <= add_s[31];
                p    <= 32'd0;
                mr   <= m;
                shc  <= sh;
                rbit <= 1'b0;
                sat  <= 1'b0;
                lost <= 1'b0;
            end
        end else if (state == Q_MUL) begin
            mr <= mr >> 1;
            if (right) begin
                p    <= add_s[32:1];
                rbit <= add_s[0];
                shc  <= shc - 5'd1;
            end else begin
                p    <= add_s[31:0];
                sat  <= sat | add_s[32] | (mr[0] & lost);
                lost <= lost | mc[31];
                mc   <= mc << 1;
            end
        end
    end

    assign idle = load;
    assign wr   = (state == Q_WR);
    always_comb begin
        if (neg)                                  q = 8'd0;
        else if (sat || p[31:8] != 24'd0 || p[7:0] == 8'hFF) q = 8'hFF;
        else                                      q = p[7:0] + {7'd0, rbit};
    end
endmodule
