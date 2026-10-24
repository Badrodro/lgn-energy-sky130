# Paper plan — reproducible, including negative results (draft 2026-10-23)

Candidates C1/C2 (see `RECHERCHE_PHASES_1-4.md`) are not part of this paper.

## Working title
"Where does the energy go in logic-gate neural networks? An open, activity-annotated, end-to-end
cost accounting on SKY130"

## Claims (each must map to a reproducible artefact)
1. An open flow from a trained DLGN to an activity-annotated ASIC energy and area breakdown:
   training → netlist → synthesis → placement/routing → gate-level simulation on real test data →
   VCD power.
2. On ASIC, the periphery (input buffering, GroupSum head, argmax) dominates area and energy.
   - This confirms FPGA observations (LILogicNet, BitLogic, DWN) and quantifies them in µm² and pJ.
   - Presented as confirmation, not discovery.
3. Negative results with fair baselines:
   - wire-length-penalised connectivity vs free placement;
   - fewer voters per class;
   - MAJ3 vs LUT2 per area;
   - number of thermometer thresholds at small scale.
4. Formal equivalence between the trained model and the final netlist (all inputs), illustrating
   a verification property specific to Boolean classifiers.

## Not claimed
- No new architecture.
- Not "first DLGN on silicon" (false, see `SILICON_STATUS.md`).
- No comparison against commercial NPUs.

## Experimental protocol
| item | setting |
|---|---|
| datasets | MNIST, Fashion-MNIST (official IDX, content hash checked); optional: 16×16 down-sampled MNIST for the silicon comparison |
| models | learned connectivity (LILogicNet/Mommen-style recipe), widths 2k / 4k / 8k gates, 1-2 layers; ≥ 3 seeds; accuracy mean ± std |
| baselines | same gate budget and training budget; head variants (adder tree, Wallace); periphery variants (parallel register vs clock-gated buffer) |
| hardware | SKY130 hd, TT corner, 1.8 V; Yosys → OpenROAD (full route, SPEF) → OpenSTA power from a VCD of SDF-annotated gate-level simulation on 1000 test images |
| metrics | accuracy; area per block (input / logic / head / argmax); energy per inference per block; leakage; f_max from STA; glitch share (zero-delay vs SDF) |
| verification | ABC `cec` equivalence of the final netlist with the reference model; gate-level simulation matches the software model bit-exactly on the full test set |

## Results to produce (status)
| table/figure | status |
|---|---|
| T1 accuracy vs gates, 3 seeds | to redo with the SOTA-level recipe (current: single seed, under-trained) |
| T2 area breakdown per block | partial (synthesis only, 2000 gates) → redo with routing |
| T3 energy breakdown per block (VCD) | partial (placement parasitics, unit delay, no clock gating) → redo |
| F1 zero-delay vs SDF energy (glitches) | to do |
| T4 negative results | done, single seed → 3 seeds |
| T5 equivalence-check runtime and result | to do |
| (paper 2) silicon vs prediction | after Phase A of `DEMONSTRATEUR.md` |

## Threats to validity (to be written in the paper)
- One PDK and corner.
- Educational open PDK with OpenRAM-class memories.
- Small tasks (MNIST-class).
- Energy from simulation, not silicon (until paper 2).
- Input encoding done off-chip.
- Results may not transfer to convolutional LGNs.

## Reproducibility artefact
- Everything in this repository: scripts, configs with seeds, `MANIFEST.sha256`, `ENVIRONMENT.txt`,
  dataset rebuild with content hash.
- Released with the paper.

## Candidate venues (deadlines to verify)
- Workshop on Open-Source EDA Technology (WOSET).
- Open-source silicon conferences (Latch-Up, ORConf), as a talk.
- IEEE Embedded Systems Letters or IEEE TCAS-II (brief).
- A design-automation conference for the full version.
