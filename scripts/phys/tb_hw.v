`timescale 1ns/1ps
module tb;
  reg clk = 0, we = 0; reg [6:0] waddr = 0; reg [31:0] wdata = 0; wire [3:0] cls;
  reg [31:0] mem [0:50*74-1]; integer n, w;
  lgn_hw dut (.clk(clk), .we(we), .waddr(waddr), .wdata(wdata), .cls(cls));
  always #10 clk = ~clk;
  initial begin
    $readmemh("img.hex", mem);
    @(negedge clk);
    $dumpfile("act.vcd"); $dumpvars(0, tb.dut);
    for (n = 0; n < 50; n = n + 1) begin
      for (w = 0; w < 74; w = w + 1) begin we = 1; waddr = w; wdata = mem[n*74+w]; @(negedge clk); end
      we = 0; @(negedge clk); @(negedge clk); @(negedge clk);
      $display("img %0d cls %0d", n, cls);
    end
    $finish;
  end
endmodule
