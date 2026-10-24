# Experiments (frozen 2026-10-23)

Everything needed to reproduce the experiments of October 2026.

| file / folder | contents |
|---|---|
| `AUDIT.md` | claim-by-claim audit (evidence, reproducibility status, fairness, caveats) |
| `ENVIRONMENT.txt` | exact tool versions |
| `MANIFEST.sha256` | SHA-256 of the large artefacts kept outside git (datasets, trained models `*.pt`) |
| `scripts/` | all scripts, with paths relative to the script (MNIST / Fashion-MNIST in `scripts/proto/dl/`, see `data.py`) |
| `notes/` | research notes: literature synthesis and candidate architectures (`RECHERCHE_PHASES_1-4.md`, in French), contribution summary, paper and demonstrator plans, measured SKY130 cost model, silicon status of logic-gate networks |
| `dtn/` | dominance-tree network engine (negative result): theory, specification, RTL, test vectors, demo program |
| `configs/` | one JSON line per run (model, sizes, epochs, `seed`, `wseed`, thresholds, ...) |
| `results/` | raw result files (`*.jsonl`), logs and summaries of synthesis, placement and power |

## 1. Setup

```sh
cd scripts/proto
python3 get_data.py      # downloads the official IDX files, checks bit-identical content
```

- The 8x8 `digits` set is scikit-learn's `load_digits()`, loaded by `data.py` (no download).
- The EDA scripts (`scripts/phys`, `scripts/head`) read the PDK from
  `/opt/asic/pdk/ciel/sky130/versions/8afc8346a57fe1ab7934ba5a6056ea8b43078e71`; replace this prefix
  with your own sky130 PDK root.

- Python: torch (CPU), numpy, scipy, scikit-learn.
- EDA: Yosys, OpenROAD/OpenSTA, Icarus Verilog, ngspice, SKY130 PDK. Versions in `ENVIRONMENT.txt`.

Seeds: every config carries `seed` (training, default 0) and, when relevant, `wseed` (fixed wiring,
default 1). Training runs used a single seed; see AUDIT.md for the consequences.

## 2. Experiment index

| topic | script | configs | results |
|---|---|---|---|
| dense vs dominance-tree networks (weight-read cost model) | `proto/dtn_exp.py` | `configs/sweep4.txt` | `results/proto/sweep4_results.jsonl` |
| quantised gaps, box dominance, delta children, wide trees, 8/4/2-bit dense, pixel reordering | `proto/dtn_q.py` | `dq.txt`, `dw.txt`, `dd.txt`, `lb.txt` | `dq_/dw_/dd_/lb_results.jsonl` |
| max-plus (tropical) layers | `proto/trop.py` | `tr.txt` | `tr_results.jsonl` |
| fixed-wiring logic nets (random / local / butterfly) | `proto/lgn_np.py`, `lgn_conv.py` | `lg1.txt`, `lg2.txt` | `lg_results.jsonl` (lg2/lg3 interrupted) |
| learned connectivity, LUT2 vs MAJ3 | `proto/lgf.py` | `h2.txt` | `h2_results.jsonl` |
| wire-length penalty and voters per class | `proto/lgf.py` | `r1.txt` | `r1_results.jsonl` |
| more datasets / sizes | `proto/lgf.py` | `m1.txt` | `m1_results.jsonl` |
| fixed-threshold baselines (fairness check) | `proto/lgf.py` | `fair.txt` | `fair_results.jsonl` |
| certification-aware training | `proto/lgfc.py` | inline (EXPERIMENTS.md §3) | `cert_results.jsonl` |
| certification during acquisition (3-valued simulation) | `proto/ramp_cert.py`, `sar_cert.py` | trained models | `audit_ramp_cert.jsonl` |
| vote margins | `proto/margins.py` | trained models | `audit_margins.jsonl` |
| hardwired netlist generation | `proto/gen_hard.py` | trained models | `results/phys/` |
| synthesis, placement, gate-level VCD power | `phys/place.tcl`, `phys/power_vcd.tcl`, `phys/tb_hw.v` | — | `results/phys/*.summary.txt`, `simA.log` |
| GroupSum head references (adder tree, Wallace) | `head/gen_head.py`, `head/tb_head.v`, `head/pw_*.tcl` | — | `results/head/` |
| comparator offset Monte-Carlo (ngspice, tt_mm) | `spice/sa.cir`, `spice/mc.sh` | — | `results/spice/mc_big.txt` |

## 3. Commands for results that were only printed to the console

```sh
# K candidate wires per gate input (L4 in AUDIT.md)
python3 lgk.py '{"data":"mnist","width":8000,"depth":1,"K":1,"epochs":5,"threads":1}'
python3 lgk.py '{"data":"mnist","width":8000,"depth":1,"K":16,"epochs":5,"threads":1}'
# certification-aware training (MNIST)
python3 lgfc.py '{"data":"mnist","widths":[2000],"kind":"lut2","epochs":10,"cert":1.0,"save":"C_cert1.pt"}'
# learned comparators vs thermometer (digits, seeds 0 and 1)
python3 lgf.py '{"data":"digits","widths":[1000],"kind":"lut2","epochs":30,"threads":1,"seed":0,"cmp":192}'
```
