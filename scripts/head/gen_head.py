import sys, math
N, K = 2000, 10; m = N // K; W = math.ceil(math.log2(m + 1))
def header(name):
    return [f"module {name} (input logic clk, input logic [{N-1}:0] v, output logic [3:0] cls);"]
def argmax_tail(L):
    L += [f"  logic [{W-1}:0] sq [{K}];", f"  always_ff @(posedge clk) for (int k = 0; k < {K}; k++) sq[k] <= s[k];",
          f"  always_ff @(posedge clk) begin logic [3:0] bc; logic [{W-1}:0] bv; bc = '0; bv = sq[0];",
          f"    for (int k = 1; k < {K}; k++) if (sq[k] > bv) begin bv = sq[k]; bc = 4'(k); end", "    cls <= bc; end", "endmodule"]
    return L
# R1a: balanced tree of W-bit adders
L = header('head_tree') + [f"  logic [{W-1}:0] s [{K}];"]; c = [0]
for k in range(K):
    t = [f"{W}'(v[{i}])" for i in range(k, N, K)]
    while len(t) > 1:
        nt = []
        for i in range(0, len(t) - 1, 2):
            c[0] += 1; L.append(f"  wire [{W-1}:0] p{c[0]} = {t[i]} + {t[i+1]};"); nt.append(f"p{c[0]}")
        if len(t) % 2: nt.append(t[-1])
        t = nt
    L.append(f"  assign s[{k}] = {t[0]};")
open('head_tree.sv', 'w').write("\n".join(argmax_tail(L)) + "\n")
# R1b: Wallace/Dadda carry-save tree of full adders, then one final adder
L = header('head_csa') + [f"  logic [{W-1}:0] s [{K}];"]; c = [0]
for k in range(K):
    cols = {0: [f"v[{i}]" for i in range(k, N, K)]}
    for wgt in range(W + 2):
        col = cols.get(wgt, [])
        while len(col) > 2:
            a, b, d = col.pop(), col.pop(), col.pop(); c[0] += 1; s_, cy = f"fs{c[0]}", f"fc{c[0]}"
            L.append(f"  wire {s_} = {a} ^ {b} ^ {d}; wire {cy} = ({a} & {b}) | ({a} & {d}) | ({b} & {d});")
            col.insert(0, s_); cols.setdefault(wgt + 1, []).append(cy)
        cols[wgt] = col
    r0 = " , ".join([(cols.get(w_, []) + ["1'b0"])[0] for w_ in reversed(range(W))])
    r1 = " , ".join([(cols.get(w_, []) + ["1'b0", "1'b0"])[1] for w_ in reversed(range(W))])
    L.append(f"  assign s[{k}] = {{{r0}}} + {{{r1}}};")
open('head_csa.sv', 'w').write("\n".join(argmax_tail(L)) + "\n")
print("ok W=", W)
