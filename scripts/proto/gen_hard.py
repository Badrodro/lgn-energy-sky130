"""Trained lgf.py model (lut2) -> hardwired SystemVerilog netlist + per-input activity file.
Usage: python3 gen_hard.py model.pt out_prefix
Writes out_prefix.sv (module lgn_hw: x -> cls, registered output) and out_prefix_act.tcl
(set_power_activity for every input port, from consecutive MNIST test images).
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import sys, json, torch, numpy as np
sys.path.insert(0, HERE + '')
from data import load

def expr(tt, a, b):
    t00, t01, t10, t11 = tt
    m = {(0,0,0,0): "1'b0", (1,1,1,1): "1'b1", (0,0,1,1): a, (0,1,0,1): b, (1,1,0,0): f"~{a}", (1,0,1,0): f"~{b}",
         (0,0,0,1): f"{a} & {b}", (0,1,1,1): f"{a} | {b}", (0,1,1,0): f"{a} ^ {b}", (1,0,0,1): f"~({a} ^ {b})",
         (1,1,1,0): f"~({a} & {b})", (1,0,0,0): f"~({a} | {b})", (0,0,1,0): f"{a} & ~{b}", (0,1,0,0): f"~{a} & {b}",
         (1,0,1,1): f"{a} | ~{b}", (1,1,0,1): f"~{a} | {b}"}
    return m[(t00, t01, t10, t11)]

ck = torch.load(sys.argv[1], weights_only=False); cfg = ck['cfg']; sd = ck['net']; out = sys.argv[2]
thr = cfg.get('thr', [0.25, 0.5, 0.75]); widths = cfg['widths']
Xtr, ytr, Xte, yte = load(cfg['data']); n_in = len(thr) * Xte.shape[1]
NWD = (n_in + 31) // 32; AWD = int(np.ceil(np.log2(NWD)))
lines = [f"module lgn_hw (input logic clk, input logic we, input logic [{AWD-1}:0] waddr, input logic [31:0] wdata, output logic [3:0] cls);",
         f"  logic [{NWD*32-1}:0] xr;",
         f"  always_ff @(posedge clk) if (we) xr[waddr*32 +: 32] <= wdata;",
         f"  wire [{n_in-1}:0] x = xr[{n_in-1}:0];"]
prev = 'x'
for li, w in enumerate(widths):
    c = sd[f'{li}.c']; t = sd[f'{li}.t']
    src = c.argmax(1).reshape(w, 2); tt = (t > 0).int().tolist()
    lines.append(f"  logic [{w-1}:0] y{li};")
    for g in range(w):
        a, b = f"{prev}[{int(src[g,0])}]", f"{prev}[{int(src[g,1])}]"
        lines.append(f"  assign y{li}[{g}] = {expr(tuple(tt[g]), a, b)};")
    prev = f"y{li}"
n = widths[-1]; G = n // 10; W = int(np.ceil(np.log2(G + 1)))
# balanced adder tree per class (explicit wires, so synthesis does not see an 800-deep chain)
lines.append(f"  logic [{W-1}:0] s [10], sq [10];")
cnt = [0]
def tree(terms, width):
    while len(terms) > 1:
        nxt = []
        for i in range(0, len(terms) - 1, 2):
            cnt[0] += 1; nm = f"p{cnt[0]}"
            lines.append(f"  wire [{W-1}:0] {nm} = {terms[i]} + {terms[i+1]};"); nxt.append(nm)
        if len(terms) % 2: nxt.append(terms[-1])
        terms = nxt
    return terms[0]
for k in range(10):
    root = tree([f"{W}'({prev}[{i}])" for i in range(k, 10 * G, 10)], W)
    lines.append(f"  assign s[{k}] = {root};")
lines += [
          "  always_ff @(posedge clk) for (int k = 0; k < 10; k++) sq[k] <= s[k];",
          f"  always_ff @(posedge clk) begin logic [3:0] bc; logic [{W-1}:0] bv; bc = '0; bv = sq[0];",
          "    for (int k = 1; k < 10; k++) if (sq[k] > bv) begin bv = sq[k]; bc = 4'(k); end",
          "    cls <= bc; end", "endmodule"]
open(out + '.sv', 'w').write("\n".join(lines) + "\n")
# input activities: one test image per clock cycle
B = np.stack([(Xte / 255.0 > tv) for tv in thr], 2).reshape(len(Xte), -1)
act = (B[1:] != B[:-1]).mean(0); duty = B.mean(0)
with open(out + '_act.tcl', 'w') as f:
    f.write("set_power_activity -input -activity 0.0 -duty 0.5\n")
    f.write(f"set_power_activity -pins [all_registers -output_pins] -activity {act.mean():.4f} -duty {duty.mean():.4f}\n")
print(json.dumps(dict(n_in=n_in, gates=sum(widths), mean_input_activity=float(act.mean()))))
