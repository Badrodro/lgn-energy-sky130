# Measured cost model (SKY130, this project)

All values below were obtained with sky130A, sky130_fd_sc_hd TT 1.80 V 25 °C,
Yosys 0.62, OpenROAD, OpenSTA, Icarus). "Measured" = extracted from the PDK liberty files or from
our own synthesis / placement / gate-level simulation runs. "Estimate" = derived by calculation.

## E.1 Primitive costs (liberty, measured)
| primitive | area µm² | leakage nW | mean input cap fF | mean internal energy / transition fJ |
|---|---|---|---|---|
| inv_1 | 3.75 | 0.0053 | 2.3 | 2.7 |
| nand2_1 | 3.75 | 0.0021 | 2.3 | 1.9 |
| and2_1 | 6.26 | 0.0027 | 1.5 | 5.0 |
| xor2_1 | 8.76 | 0.0043 | 4.4 | 7.8 |
| mux2_1 | 11.26 | 0.0040 | 2.3 | 6.1 |
| mux4_1 | 26.28 | 0.0105 | 2.1 | 9.6 |
| maj3_1 | 10.01 | 0.0021 | 2.8 | 6.8 |
| fa_1 (full adder) | 20.02 | 0.0073 | 5.8 | 12.7 |
| dlxtp_1 (latch) | 15.01 | 0.0079 | 1.8 | 14.5 |
| dfxtp_1 (flip-flop) | 20.02 | 0.0084 | 1.7 | 13.9 |
| OpenRAM 2 KB macro 32x512 (1rw1r) | 284 538 (0.285 mm², **17.4 µm²/bit**) | 17 700 | — | **13 808 per access** |

Observations (measured): in this open PDK a latch bit (15.0 µm²) is denser than a bit of the
available OpenRAM 2 KB macro (17.4 µm² incl. periphery), with ~2000x lower leakage per bit.
One 32-bit SRAM read (13.8 pJ) costs as much as ~1000 latch/gate transitions.

## E.2 Logic-gate network on MNIST (learned connectivity, 2000 gates, 1 layer, 10 epochs)
| item | value | status |
|---|---|---|
| accuracy (software, discretised) | 92.66 % | measured |
| gate-level simulation of the synthesised netlist, 50 test images | 49/50 correct | measured |
| synthesised area incl. 2368-bit input register, popcount (GroupSum), argmax | 78 099 µm² | measured |
| GroupSum popcount alone, 2000 voters (200/class), synthesised | 60 215 µm² (~5x the 2000 gates) | measured |
| power from gate-level VCD, 50 MHz, 77 cycles per inference (32-bit loads) | 2.02 mW -> 3.1 nJ / inference | measured (OpenSTA, placement parasitics, ideal clock) |
| share of sequential (input register clocked every cycle, no clock gating) | 84 % | measured |
| logic + popcount energy | ~0.5 nJ / inference | measured share |
| with per-word clock gating of the input register | ~0.5 nJ / inference | estimate |

## E.3 Refuted / weak hypotheses tested in this project (for the record)
| hypothesis | result | status |
|---|---|---|
| Dominance-tree networks (DTN) beat width-optimised dense nets in weight reads | tie at best on digits / MNIST / F-MNIST | measured |
| Max-plus (tropical) layers with sorted-scan evaluation | 95.3 % @ 950 words vs dense 4-bit 95.2 % @ 735 | measured |
| Learned pairwise analog comparators replace thermometer inputs | -7 points at equal count (digits) | measured |
| MAJ3 gates beat LUT2 per area | 93.1 % vs 92.7 % at equal count, but MAJ3 is 1.6x larger | measured |
| Fewer voters per class to shrink GroupSum | -7 to -11 points (20 or 5 voters / class) | measured |
| Wire-length-penalised connectivity shortens real wires | after free placement: 1.7x LONGER HPWL, 1.4x larger area | measured (refuted) |
| Fixed butterfly wiring as universal programmable fabric | trains like random wiring, but fixed wiring needs ~50x more gates than learned wiring (literature) | measured + literature |
