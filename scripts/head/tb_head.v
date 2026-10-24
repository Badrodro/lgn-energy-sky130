`timescale 1ns/1ps
module tb;
  reg clk = 0; reg [2015:0] vv; reg [31:0] mem [0:200*63-1]; integer n, w; wire [3:0] cls;
  `HEAD dut (.clk(clk), .v(vv[1999:0]), .cls(cls));
  always #10 clk = ~clk;
  initial begin
    $readmemh("voters.hex", mem); vv = 0;
    @(negedge clk); $dumpfile(`VCD); $dumpvars(0, tb.dut);
    for (n = 0; n < 200; n = n + 1) begin
      for (w = 0; w < 63; w = w + 1) vv[32*w +: 32] = mem[n*63+w];
      @(negedge clk);
      if (n >= 2) $display("cls %0d", cls);
    end
    @(negedge clk); $display("cls %0d", cls); @(negedge clk); $display("cls %0d", cls);
    $finish;
  end
endmodule
