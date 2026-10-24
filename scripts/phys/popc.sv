module popc #(parameter int N = 2000) (input logic clk, input logic [N-1:0] y, output logic [3:0] cls);
  localparam int G = N / 10, W = $clog2(G + 1);
  logic [W-1:0] s [10], sq [10];
  always_comb for (int c = 0; c < 10; c++) begin s[c] = '0; for (int i = c; i < N; i += 10) s[c] += W'(y[i]); end
  always_ff @(posedge clk) for (int c = 0; c < 10; c++) sq[c] <= s[c];
  always_ff @(posedge clk) begin logic [3:0] bc; logic [W-1:0] bv; bc = '0; bv = sq[0];
    for (int c = 1; c < 10; c++) if (sq[c] > bv) begin bv = sq[c]; bc = 4'(c); end
    cls <= bc; end
endmodule
