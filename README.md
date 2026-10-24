# What Does a Logic-Gate Neural Network Really Cost on a Chip?

An open, reproducible measurement of the area and energy of logic-gate neural networks in the SkyWater SKY130 process, counting the whole circuit: the logic gates, but also the input register, the vote-counting output head and the wires.

## In one minute

A *logic-gate network* (LGN) is a neural network where every neuron is a tiny two-input Boolean gate (AND, XOR, ...) learned during training. Papers present LGNs as extremely efficient because the gates themselves are tiny. But a chip also has to **hold the input image** and **count the votes** of the last layer to pick the class, and those parts are usually left out of the comparison.

This project builds the complete circuit and measures it:

```
trained model ──> Verilog netlist ──> synthesis (Yosys) ──> placement (OpenROAD)
              ──> gate-level simulation on real MNIST images ──> power from switching activity
```

The hardware netlist classifies the test images exactly like the software model (49/50 correct in gate-level simulation, the same answers as Python).

**Main finding:** on chip, the parts around the network dominate. 84 % of the power goes into flip-flops, mostly the 2,368-bit input register, and the vote-counting head alone is about 60,000 µm², while each logic gate is about 12 µm².

## Measured results (SKY130, typical corner)

| What | Result | Source |
| :--- | :--- | :--- |
| Full design: 2,000 gates + 2,368-bit input register + output head | 78,099 µm², 3.1 nJ per inference at 50 MHz | `results/phys/pl_A_wl0.summary.txt`, `pwA.summary.txt`, `simA.log` |
| Where the power goes | 84 % in flip-flops (input register clocked every cycle, no clock gating) | `results/phys/pwA.summary.txt` |
| Output head that counts 2,000 votes into 10 classes | 61,969 µm² (adder tree) / 60,111 µm² (Wallace tree), exact on 199/199 test vectors | `results/head/` |
| Output head energy | 71 / 68 pJ per decision (lower bound, see limitations) | `results/head/pw_*.summary.txt` |
| One logic gate | about 12 µm² | `results/phys/dlgn_gates.summary.txt` |

## Ideas that did not work (negative results)

Each was compared against a fair baseline with the same budget. They are published so that nobody has to try them again blindly.

- **Penalising wire length during training does not shorten real wires.** After placement, total wire length went up (0.47 m vs 0.28 m) and area grew to 107,501 µm² (vs 78,099 µm²).
- **Fewer voters per class costs accuracy.** 200 / 20 / 5 voters per class give 92.7 / 85.4 / 81.2 % on MNIST.
- **3-input majority gates are not better than 2-input gates** at equal gate count (93.1 vs 92.7 %).
- **Several input thresholds ≈ one threshold** at 2,000 gates (92.5 vs 92.7 % on MNIST).
- **Dominance-tree networks** (`dtn/`): neurons arranged so that an inactive "parent" proves its "children" inactive, letting the hardware skip their weights exactly. The RTL engine works and is tested, but the scheme at best ties with a simply narrower dense network.
- **Stopping the sensor acquisition early once the answer is certain** brings no gain at this scale.

One idea is reported as an **estimate only**: deciding the class in analog when the vote margin is large and recomputing digitally only the close cases. It could save roughly 2× area and 5× energy on the output head, but it was not simulated at circuit level (`scripts/spice/`, `scripts/proto/margins.py`).

## Limitations

- Single training seed. Accuracy is below the state of the art for LGNs (the literature reports 98.45 % on MNIST with 8,000 gates).
- Power uses pin capacitances and placement estimates, not routed wire parasitics, and a unit-delay simulation that misses most glitches. **Energy figures are lower bounds.**
- No clock gating in the input register.
- Full list, claim by claim: [`AUDIT.md`](AUDIT.md).

## Reproduce

```sh
cd scripts/proto
python3 get_data.py          # downloads MNIST / Fashion-MNIST, checks bit-identical content
```

Then follow [`EXPERIMENTS.md`](EXPERIMENTS.md): one row per experiment, with script, config and result file. Trained models (`*.pt`, 567 MB) are not included; retrain them from `configs/` (seeds included) and check them against `MANIFEST.sha256`. Tool versions are in `ENVIRONMENT.txt`. The EDA scripts expect a SKY130 PDK; edit the PDK path as explained in `EXPERIMENTS.md`.

| Folder | Contents |
| :--- | :--- |
| `scripts/proto/` | training and analysis (Python, PyTorch) |
| `scripts/phys/`, `scripts/head/` | netlists, synthesis, placement, gate-level power (Yosys, OpenROAD, OpenSTA, Icarus Verilog) |
| `scripts/spice/` | comparator offset Monte-Carlo (ngspice) |
| `dtn/` | dominance-tree engine: specification, theory, SystemVerilog RTL, test vectors |
| `configs/`, `results/` | run configurations and raw results |
| `notes/` | research notes (partly in French) |

## How this was made

I use AI assistants (Claude) in my projects; I choose the questions and check the results.

## Author

Badreddine Benfettouma, undergraduate in electronics at Université Bourgogne Europe. Website: [badrodro.github.io](https://badrodro.github.io). License: Apache-2.0.
