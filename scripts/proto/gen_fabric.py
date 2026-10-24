"""Generate the SystemVerilog of a programmable logic-gate fabric with FIXED wiring.
  gate      = sky130 mux4 (S1 = a, S0 = b) whose 4 data inputs are configuration latches
  config    = 32-bit words written through a simple port (8 gates per word), one latch enable per word
  output    = per-class popcount of the last layer (gate i -> class i mod 10) and argmax
Usage: python3 gen_fabric.py <json spec> <out.sv>
  spec: {"in_shape":[H,W,C], "local":[[h,w,c],...], "r":2, "wseed":1}  or  {"n_in":N, "width":w, "depth":d, "wseed":1}
"""
import os as _os; HERE = _os.path.dirname(_os.path.abspath(__file__))  # data in HERE/dl
import json, sys
import numpy as np
sys.path.insert(0, HERE + '')
from lgn_np import random_wiring, local_wiring

def build(spec):
    gw = np.random.default_rng(spec.get('wseed', 1))
    if 'local' in spec:
        H, W, C = spec['in_shape']; n_in = H * W * C
        layers = [tuple(l) for l in spec['local']]
        wire, _ = local_wiring(gw, (H, W, C), layers, spec.get('r', 2), spec.get('bal', False)); widths = [h * w * c for h, w, c in layers]
    else:
        n_in = spec['n_in']; widths = [spec['width']] * spec['depth']
        wire, _ = random_wiring(gw, n_in, widths)
    return n_in, widths, wire

def emit(spec, path):
    n_in, widths, wire = build(spec)
    G = sum(widths); NW = (G + 7) // 8; AW = max(1, int(np.ceil(np.log2(NW))))
    nl = widths[-1]; per = nl // 10; SW = int(np.ceil(np.log2(per + 1)))
    o = []
    o.append(f"// generated: {len(widths)} layers {widths}, {G} gates, {n_in} input bits")
    o.append("module lgn_fabric (")
    o.append("    input  logic clk, input logic rst,")
    o.append(f"    input  logic cfg_we, input logic [{AW-1}:0] cfg_addr, input logic [31:0] cfg_wdata,")
    o.append(f"    input  logic [{n_in-1}:0] x,")
    o.append(f"    output logic [3:0] cls, output logic [{SW-1}:0] best")
    o.append(");")
    o.append(f"    logic [{NW*32-1}:0] cfg;")
    o.append(f"    // configuration latches: word k is transparent while cfg_we & cfg_addr == k")
    o.append(f"    genvar k;")
    o.append(f"    generate for (k = 0; k < {NW}; k++) begin : cw")
    o.append(f"        wire en = cfg_we & (cfg_addr == {AW}'(k));")
    o.append(f"        for (genvar j = 0; j < 32; j++) begin : cb")
    o.append(f"            sky130_fd_sc_hd__dlxtp_1 l (.D(cfg_wdata[j]), .GATE(en), .Q(cfg[32*k+j]));")
    o.append(f"        end")
    o.append(f"    end endgenerate")
    base = 0; prev = 'x'
    for li, ((ia, ib), w) in enumerate(zip(wire, widths)):
        o.append(f"    logic [{w-1}:0] y{li};")
        for i in range(w):
            g = base + i
            o.append(f"    sky130_fd_sc_hd__mux4_1 g{li}_{i} (.A0(cfg[{4*g}]), .A1(cfg[{4*g+1}]), .A2(cfg[{4*g+2}]), .A3(cfg[{4*g+3}]), "
                     f".S0({prev}[{ib[i]}]), .S1({prev}[{ia[i]}]), .X(y{li}[{i}]));")
        base += w; prev = f"y{li}"
    last = f"y{len(widths)-1}"
    o.append(f"    // per-class popcount (gate i -> class i mod 10), registered, then argmax")
    o.append(f"    logic [{SW-1}:0] s [10];")
    o.append(f"    always_comb for (int c = 0; c < 10; c++) begin")
    o.append(f"        s[c] = '0;")
    o.append(f"        for (int i = c; i < {nl}; i += 10) s[c] += {SW}'({last}[i]);")
    o.append(f"    end")
    o.append(f"    logic [{SW-1}:0] sq [10];")
    o.append(f"    always_ff @(posedge clk) for (int c = 0; c < 10; c++) sq[c] <= s[c];")
    o.append(f"    always_ff @(posedge clk or posedge rst)")
    o.append(f"        if (rst) begin cls <= '0; best <= '0; end")
    o.append(f"        else begin")
    o.append(f"            logic [3:0] bc; logic [{SW-1}:0] bv; bc = '0; bv = sq[0];")
    o.append(f"            for (int c = 1; c < 10; c++) if (sq[c] > bv) begin bv = sq[c]; bc = 4'(c); end")
    o.append(f"            cls <= bc; best <= bv;")
    o.append(f"        end")
    o.append("endmodule")
    open(path, 'w').write("\n".join(o) + "\n")
    return dict(gates=G, cfg_words=NW, in_bits=n_in, last=nl)

if __name__ == '__main__':
    print(json.dumps(emit(json.loads(sys.argv[1]), sys.argv[2])))
