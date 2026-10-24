# DTN engine — specification v1

Dominance-Tree Network (DTN) inference engine for the RV32I-Alpha SoC.
Author: Badreddine Benfettouma. Status: research prototype; no gain over width-optimised dense networks (see `../AUDIT.md`, L2).

## 1. Idea in one paragraph

A ReLU layer is organised as a forest: `R` roots, each with `C` children. Every child is
constrained to be *dominated* by its root: `w_child = w_root - D` with `D >= 0` element-wise,
and `b_child = b_root - db` with `db >= 0`. Activations are non-negative (pixels or ReLU
outputs), so `acc_child = acc_root - D.x - db <= acc_root`, and the output function
(requantisation, monotone non-decreasing, clamped at 0) gives `out_child <= out_root`.
Hence **an inactive root proves that all its children are inactive**: they are skipped without
reading them, and the result is still exactly the dense integer computation of the layer.
Active children are computed incrementally from the root accumulator, reading only their small
non-negative gaps `D` (NB bit-planes). Weight bit-planes are read MSB-first with a certified
early stop, and words whose 32 activations are all zero are never read.

## 2. Number formats

| Quantity | Format |
|---|---|
| activation `x` | uint8 (0..255) |
| root weight | sign-magnitude, 1 sign bit + 7 magnitude bits, value in [-127, 127]; a zero weight has sign 0 |
| child gap `D` | unsigned, `NB` bits (1 <= NB <= 8) |
| root bias `b` | int32, the compiler guarantees `|b| < 2^28` |
| child bias gap `db` | uint32, `db < 2^28` |
| accumulator | signed; with N <= 256 and the bias limits above every value below fits in 32-bit two's complement |
| requantisation | `out = clamp((acc * M + 2^(SH-1)) >> SH, 0, 255)`, `M` in [1, 2^15-1], `SH` in [1, 30], `>>` arithmetic |
| `T0` | `floor((2^(SH-1) - 1) / M)`: the largest `acc` with `out == 0` (computed by software) |

## 3. Weight memory image

32-bit words, word-addressed. `LANES = 32`: lane `i` of word `g` of a plane is input
`j = 32*g + i`, stored in bit `i` (bit 0 = lane 0). `G = ceil(N / 32)`. Bits of lanes
`j >= N` must be 0 (the engine also masks them).

A layer starts at word address `WBASE`. For `r = 0 .. R-1`, the subtree of root `r` is stored
contiguously:

```
ROOT r   : word 0           bias b (int32)
           words 1 .. 8G    planes in the order p = 7 (sign), 6, 5, 4, 3, 2, 1, 0 (magnitude),
                            plane p occupies G consecutive words (group 0 first)
CHILD r,c: (c = 0 .. C-1, right after the root record, child records back to back)
           word 0           db (uint32)
           words 1 .. NB*G  gap planes p = NB-1, ..., 0, each G words
```

`S_ROOT = 1 + 8*G`, `S_CHILD = 1 + NB*G`, `S_TREE = S_ROOT + C*S_CHILD`,
root `r` record base = `WBASE + r*S_TREE`, child `(r,c)` base = root base + `S_ROOT + c*S_CHILD`.
Root weight `w = (sign ? -1 : +1) * sum_{p=0..6} bit_p * 2^p`; child weight `w - sum_{p<NB} bit_p * 2^p`.

## 4. Activation buffers and outputs

Two buffers `BUF0`, `BUF1`, each `32*G_MAX` bytes (G_MAX is a generic, default 4 -> 128 bytes).
A layer reads `x` from buffer `IN_BUF` and writes its outputs to the other one.
Output index of root `r` is `o = r*(1+C)`, of child `(r,c)` is `o = r*(1+C) + 1 + c`;
`R*(1+C) <= 32*G_MAX`. Bytes of the output buffer at indices `>= R*(1+C)` are left unchanged.
At layer start the engine uses `x[j] = BUF[IN_BUF][j]` for `j < N` and `x[j] = 0` for `j >= N`.

## 5. Algorithm (normative: outputs AND the number of weight words read must match)

```
nz[g] = OR over the lanes of group g of (x[j] != 0)          for g < G
S     = sum_j x[j]
WORDS = 0                                                      # weight words read in this layer
for r in 0 .. R-1:
    base = WBASE + r*S_TREE
    b    = W[base]; WORDS += 1                                  # the bias word is always read
    Sneg = 0
    for g in 0..G-1:                                            # sign plane
        if nz[g]: sw[g] = W[base+1+g]; WORDS += 1; Sneg += sum_i sw[g].bit(i) * x[32g+i]
        else:     sw[g] = 0
    Spos = S - Sneg
    acc  = b
    cert = (KIND == RELU) and (acc + (2^7 - 1)*Spos <= T0)
    for p in 6 downto 0:                                        # magnitude planes, MSB first
        if cert: break
        for g in 0..G-1:
            if nz[g]:
                mw = W[base + 1 + (7-p)*G + g]; WORDS += 1
                acc += 2^p * sum_i mw.bit(i) * (sw[g].bit(i) ? -x[32g+i] : +x[32g+i])
        cert = (KIND == RELU) and (acc + (2^p - 1)*Spos <= T0)
    # here: cert, or every plane was read and acc is exact
    if KIND == RELU:
        out_root = 0 if cert else requant(acc)
        BUF_out[r*(1+C)] = out_root
        for c in 0 .. C-1:
            o = r*(1+C) + 1 + c
            if cert: BUF_out[o] = 0; continue                   # skipped: dominance lemma
            cb = base + S_ROOT + c*S_CHILD
            db = W[cb]; WORDS += 1
            a  = acc - db
            ccert = (a <= T0)
            for p in NB-1 downto 0:
                if ccert: break
                for g in 0..G-1:
                    if nz[g]:
                        dw = W[cb + 1 + (NB-1-p)*G + g]; WORDS += 1
                        a -= 2^p * sum_i dw.bit(i) * x[32g+i]
                ccert = (a <= T0)
            BUF_out[o] = 0 if ccert else requant(a)
    else:  # KIND == ARGMAX: C must be 0, no early stop, acc exact for every row
        if r == 0 or acc > best: best = acc; idx = r            # first maximum wins
RESULT = idx; MAXACC = best (ARGMAX only)
```

Notes:
- When all planes have been read, `cert == (acc <= T0) == (requant(acc) == 0)`.
- A child is only computed when its root is active, i.e. when `acc` is exact.
- Soundness: the unread magnitude planes of a root add at most `(2^p - 1)` per positive-weight
  lane and at least 0 (they can only make negative weights more negative), so
  `acc + (2^p - 1)*Spos` is an upper bound of the exact accumulator; the unread gap planes of a
  child can only decrease `a`.

## 6. MMIO register map (byte offsets inside the engine window)

| Offset | Name | Access | Content |
|---|---|---|---|
| 0x000 | CTRL | W | bit0 START (self-clearing), bit1 KIND (0 RELU-tree, 1 ARGMAX), bit2 IN_BUF |
| 0x004 | STATUS | R | bit0 BUSY, bit1 DONE (set at the end of a layer, cleared by START) |
| 0x008 | WBASE | RW | word address of the layer |
| 0x00C | N | RW | number of inputs, 1 .. 32*G_MAX |
| 0x010 | R | RW | number of roots (rows), 1 .. 255 |
| 0x014 | C | RW | children per root, 0 .. 15 |
| 0x018 | NB | RW | gap planes, 1 .. 8 |
| 0x01C | REQ_M | RW | M (15 bits) |
| 0x020 | REQ_SH | RW | SH (5 bits) |
| 0x024 | T0 | RW | int32 |
| 0x028 | RESULT | R | argmax index |
| 0x02C | MAXACC | R | argmax accumulator (int32) |
| 0x030 | WORDS | R | weight words read by the last layer |
| 0x034 | CYCLES | R | clock cycles of the last layer |
| 0x400 .. 0x7FF | BUF0 | RW | activation buffer 0, byte addressable (32-bit accesses with byte enables) |
| 0x800 .. 0xBFF | BUF1 | RW | activation buffer 1 |

Writes to the buffers or registers while BUSY are ignored. Reads return the register or the
4 bytes of the addressed word (little endian, byte `4k` in bits 7:0).

## 7. SystemVerilog module (rtl/dtn/dtn_engine.sv)

```systemverilog
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
    input  logic [11:0]      s_addr,    // byte offset in the window
    input  logic [31:0]      s_wdata,
    output logic [31:0]      s_rdata,   // valid in the cycle after the request
    // weight memory read port (synchronous, data valid in the cycle after w_en, not held after)
    output logic             w_en,
    output logic [AW-1:0]    w_addr,
    input  logic [LANES-1:0] w_rdata,
    output logic             done       // = STATUS.DONE
);
```
With `LANES < 32` (formal only) a "word" has LANES bits and groups have LANES lanes; every
formula above holds with 32 replaced by LANES. The activation buffers then hold LANES*G_MAX
bytes each, at the same MMIO offsets.
Coding rules: the SystemVerilog subset accepted by Yosys 0.33, Icarus 13, Verilator 5 and
Vivado (logic, always_ff, always_comb, packed arrays, functions, generate; no interfaces,
no structs/unions, no classes); Verilator `--lint-only -Wall` clean; no initial values; async
reset only on control registers; ASIC timing hint: `w_addr`/`w_en` should come from registers
or very short logic.

## 8. Test vector format (for the GHDL testbench and the golden model)

Plain text, one command per line, `#` starts a comment, numbers in hex without prefix
unless stated:
```
W <addr> <data>                    write weight word (word address, 32-bit data)
A <buf> <index> <byte>             write activation byte into buffer 0/1
L <kind> <inbuf> <wbase> <N> <R> <C> <NB> <M> <SH> <T0>
                                   run one layer (all fields hex, T0 as 32-bit two's complement)
O <buf> <index> <byte>             expect this activation byte after the last L
R <idx>                            expect RESULT (ARGMAX)
X <maxacc>                         expect MAXACC (32-bit two's complement)
N <words>                          expect WORDS of the last L
```

## 9. Properties to prove formally (SymbiYosys, small instance e.g. LANES=4, G_MAX=1)

1. Functional equivalence: for every weight memory content and every activation vector
   (symbolic), the outputs written by one RELU-tree layer with R=1, C<=2, NB<=8 equal the
   reference integer computation (dense dot products, requantisation), and RESULT/MAXACC equal
   the reference for ARGMAX with R<=2.
2. WORDS equals the number of cycles with `w_en = '1'` during the layer.
3. The engine never reads a weight word of a skipped child or of a group with `nz[g] = 0`.
4. Liveness: a started layer reaches DONE within a bound that depends only on R, C, NB, G.
