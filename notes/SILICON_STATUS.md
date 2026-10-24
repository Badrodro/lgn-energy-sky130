# Has a differentiable logic gate network (DLGN) been measured on silicon? — status 2026-10-23

Scope: Petersen-style **learned logic gate networks** (DLGN, convolutional DLGN, connection-optimised
LGN, LILogicNet).

Other logic-based classifiers, listed separately because they are **different model families**:
- Tsetlin machines;
- evolved tiny classifier circuits;
- LUT/weightless networks on FPGA;
- binarised neural networks.

Method:
- web search (several queries);
- arXiv full texts;
- Tiny Tapeout shuttle index (`index.tinytapeout.com/ttsky25a.json`) and its "silicon proven
  projects" page;
- project repositories.
- **Limits**: patent databases, IEEE Xplore and ACM DL were not consulted;
  conference talks and private results are not covered.

## Evidence ladder for DLGN-family hardware

| level | reached? | evidence |
|---|---|---|
| software / GPU-CPU inference | yes, many | Petersen et al. NeurIPS 2022, 2024; Mommen et al. 2025/2026; LILogicNet (arXiv 2511.12340) |
| FPGA implementation (synthesis + P&R + board timing) | yes, many | Petersen 2024 (Xilinx VU13P / XC7Z045); LILogicNet (XC7A200T); BitLogic (arXiv 2602.07400) |
| ASIC synthesis / place & route / post-layout power | yes | Fieldhouse & Tang, "Silicon Aware Neural Networks", arXiv 2604.19334 (SKY130 hard macro, 97 % MNIST, 41.8 M inf/s, 83.88 mW at assumed activity 0.7; tape-out stated as future work) |
| GDSII | yes | Zioma & Armaitis: public GDS for SKY130 (Tiny Tapeout ttsky25a, project #72 `tt_um_rejunity_lgn_mnist`), IHP SG13G2 (ttihp25b, #336) and GF180 (Fashion-MNIST, gitlab `rejunity/ws0-lgn-fxnist-gf180mcu-tapeout`) |
| fabrication | **yes (multi-project shuttles)** | LILogicNet §4.3.2: "We taped out 6K-gate LILogicNet-S MNIST variant in SKY130 and SG13 … followed by an 8K-gate LILogicNet-S FashionMNIST variant in GF180"; ttsky25a development kits are sold by Tiny Tapeout, so the shuttle was delivered |
| **published silicon measurement** (accuracy, f_max, energy, leakage) | **none found** | no measurement in LILogicNet v2 (2026-04-26) or in the project repository ("How to test: TO DO"); project absent from the Tiny Tapeout "silicon proven projects" list (257 entries, including four other projects of the same author) |

## Neighbouring families with measured silicon (not DLGN)

- Mixed-signal binarised CNN: Bankman et al., JSSC 2019 (28 nm, 3.8 µJ / 86 % CIFAR-10).
- Tsetlin-machine ASIC, 65 nm, ~8.6 nJ/frame (register C, entry L53, metadata level).
- Evolved "tiny classifier" circuits on flexible ICs (Nature Electronics 2024; register C).
  To be re-checked at full-text level before any comparison.

## Correct formulation for any publication

> As of October 2026, DLGN-family classifiers have been implemented on FPGA, laid out in SKY130
> (post-layout simulation), and taped out on multi-project shuttles (SKY130, IHP SG13G2, GF180) by
> Zioma and Armaitis, but we found no published silicon measurement of accuracy, maximum frequency
> or energy.

Do **not** claim "first DLGN on silicon" (false), nor "first measured DLGN" unless the claim is
re-checked immediately before submission. The chips exist and their designers may publish first.

## Consequence for the demonstrator

The fastest credible silicon evidence is to **characterise the existing open-source chip**:
- Tiny Tapeout ttsky25a development kit, project #72 (MNIST 16×16, 1 MHz nominal clock).
- Ideally in collaboration with, or with acknowledgement of, its designers.
- Then compare with a design of our own on a later shuttle. See `DEMONSTRATEUR.md`.
