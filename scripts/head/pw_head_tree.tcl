read_liberty /opt/asic/pdk/ciel/sky130/versions/8afc8346a57fe1ab7934ba5a6056ea8b43078e71/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog head_tree_syn.v
link_design head_tree
create_clock -period 20 [get_ports clk]
read_vcd -scope tb/dut head_tree.vcd
report_power
