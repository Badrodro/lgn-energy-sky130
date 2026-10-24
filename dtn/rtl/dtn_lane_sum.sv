// DTN engine: masked lane sum.
// Engineer: Badreddine BENFETTOUMA (University of Bourgogne Europe)
//
// sum = sum over the LANES lanes i of (en[i] ? x[i] : 0), x[i] = byte i of x (unsigned).
// This is the only wide datapath of the engine. It is a balanced binary tree of adders
// (LANES - 1 adders, 5 levels for 32 lanes); every lane is unsigned, so no sign handling.
// LANES must be a power of two (32 in silicon, 4 for formal).
//
// The tree is stored heap-like in t (node k in bits SW*k + SW-1 .. SW*k): the leaves are the
// nodes LANES-1 .. 2*LANES-2, node k adds its children 2k+1 and 2k+2, the root is node 0.
// All nodes are SW bits wide; the upper bits of the lower levels are constant zero and are
// removed by synthesis.

module dtn_lane_sum #(
    parameter int LANES = 32
) (
    input  logic [8*LANES-1:0]         x,     // lane i in bits 8i+7 .. 8i
    input  logic [LANES-1:0]           en,    // lane enables
    output logic [8+$clog2(LANES)-1:0] sum
);
    localparam int SW = 8 + $clog2(LANES);

    logic [SW*(2*LANES-1)-1:0] t;       // flat: Yosys 0.33 has no packed 2-D arrays

    for (genvar i = 0; i < LANES; i++) begin : g_leaf
        assign t[SW*(LANES-1+i) +: SW] = {{(SW-8){1'b0}}, x[8*i +: 8] & {8{en[i]}}};
    end

    for (genvar k = 0; k < LANES-1; k++) begin : g_node
        assign t[SW*k +: SW] = t[SW*(2*k+1) +: SW] + t[SW*(2*k+2) +: SW];
    end

    assign sum = t[SW-1:0];
endmodule
