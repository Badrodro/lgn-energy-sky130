# Audit of measurements and comparisons

Audit date: 2026-10-23. Environment: `ENVIRONMENT.txt`.
Artefacts not stored in git (datasets, trained models): `MANIFEST.sha256`.
Datasets: `scripts/proto/get_data.py` rebuilds MNIST / Fashion-MNIST from the official IDX files
and checks **bit-identical array content** (verified on 2026-10-23).

Legend:
- Status: **R** = reproduced during the audit (same numbers); **D** = deterministic artefact
  (synthesis/simulation), not re-run; **S** = single run, not re-run.
- Fairness: **OK** = baseline equal in budget and evaluated in the same flow; **WEAK** = see note;
  **NO** = comparison must not be published as is.

## 1. Learning-side results

| # | claim | evidence (results/...) | status | fairness | caveats |
|---|---|---|---|---|---|
| L1 | 4-bit dense + pixel reordering is the "weight-read" frontier on MNIST (97.2 % at ~1.4 k words) | proto/lb_results.jsonl | S | WEAK | analytical cost model (32-bit words read, MSB-first early termination), not hardware; seed 0; 10 epochs |
| L2 | DTN trees tie at best with width-optimised dense (MNIST/F-MNIST) | proto/sweep4_results.jsonl, dq/dw/dd_results | S (digits: 3 seeds) | OK within the cost model | same caveat as L1 |
| L3 | max-plus layers do not beat 4-bit dense | proto/tr_results.jsonl | S | WEAK | 12 epochs only, LSE relaxation; readout cost model analytical |
| L4 | learned connectivity >> fixed wiring (K=1: 83.9 %, K=16: 93.6 %, 1x8000, 5 epochs) | console only; command in README section 3 | S | OK | to be re-run with a result file before publication |
| L5 | MAJ3 ≈ LUT2 at equal gate count (93.1 vs 92.7 %, MNIST, 2000) | proto/h2_results.jsonl | S | OK | 1 seed |
| L6 | fewer voters per class lose accuracy (200/20/5 voters: 92.7 / 85.4 / 81.2 %) | proto/r1_results.jsonl | S | OK | 1 seed; with 5 voters ~12 % of test images end in exact ties (tie rule = lowest index) |
| L7 | thermometer thresholds add little at 2000 gates (MNIST 1 thr 92.5 vs 3 thr 92.7; F-MNIST 83.3 / 83.4 / 83.5 for 1/2/3 thr) | proto/fair_results.jsonl, m1_results.jsonl | S | OK | differences below expected seed noise; needs ≥3 seeds |
| L8 | our LGN accuracies are **below the state of the art** (e.g. 8000 gates: literature 98.45 % MNIST vs our 10-epoch recipe) | LILogic Net (Fojcik et al. 2025, arXiv:2511.12340) | — | — | **any efficiency claim vs other model families must use SOTA-trained LGNs** |

## 2. Certification analyses (deterministic given a model)

| # | claim | evidence | status | caveats |
|---|---|---|---|---|
| C2a | fraction of decisions certified during a rising-threshold acquisition (3-valued simulation) | proto/audit_ramp_cert.jsonl (11 models) | **R** (identical to first runs) | conservative 3-valued logic (may under-certify); exact ties counted as not certified |
| C2b | C2 brings no gain over a fixed 1-threshold network at 2000 gates | fair_results + audit_ramp_cert | **R** (cert) + S (training) | negative result, valid at this scale only |
| C1a | top-1/top-2 vote margin: P(≤3 votes) = 3.0 %, P(≤5) = 4.6 % (MNIST, 2000 voters) | proto/audit_margins.jsonl (4 models) | **R** | — |

## 3. Hardware measurements (SKY130, open flow)

| # | claim | evidence | status | fairness | caveats |
|---|---|---|---|---|---|
| H1 | GroupSum head (2000 voters, 10 classes): 61 969 µm² (adder tree) / 60 111 µm² (Wallace CSA), exact (199/199 vs Python) | head/*.summary.txt, head/sim_*.log | D | OK | synthesis area only (no placement); TT corner |
| H2 | head energy 71 / 68 pJ per decision from gate-level VCD on real voter vectors | head/pw_*.summary.txt | D | WEAK | **pin capacitances only (no wire), unit-delay simulation (glitches largely missed) → lower bound**; ideal clock |
| H3 | full LGN (2000 gates) + 2368-bit input register + head: 78 099 µm²; 49/50 correct in gate-level sim; 3.1 nJ/inference at 50 MHz, 84 % in the input register | phys/*.summary.txt, phys/simA.log | D | WEAK | register clocked every cycle (no clock gating); placement parasitics (global placement, no routing extraction); ideal clock |
| H4 | wire-penalised connectivity does NOT shorten real wires (HPWL 0.47 m vs 0.28 m; area 107.5 k vs 78.1 k µm²) | phys/pl_*.summary.txt | D | OK at 2000 gates | free placement; routability at large scale (where the literature reports congestion) **not tested** |
| H5 | latch bit (15.0 µm²) denser than OpenRAM 2 KB macro bit (17.4 µm²), ~2000x less leakage per bit | liberty/LEF values (`notes/review/E_measured_sky130.md`) | D | OK | macro area includes periphery; OpenRAM macros are not production SRAM |
| H6 | StrongARM comparator (input pair 2/0.5 µm): σ_offset = 6.10 mV, 65.6 fJ/decision (100 MC samples, tt_mm) | spice/mc_big.txt | D | — | hand-sized, TT corner, 1.8 V, 100 MHz; offset resolution 0.625 mV |
| H7 | MIM mismatch σ(ΔC/C) = 2.8 % / √(area µm²) | PDK model file | D | — | model value, not silicon |

## 4. Comparisons that must NOT be published as is

1. "LGN 0.5 nJ vs SRAM engine 19 nJ": the SRAM figure counts SRAM reads only (analytical), the LGN
   figure is VCD-measured and excludes wires. **Different methodologies.** Either measure both in the
   same flow or drop.
2. "LGN ~10x more efficient than our MLP engine": same issue, plus LGN accuracy below SOTA (L8).
3. C1 (analog certified head) energy/area: **estimates only**, no circuit-level simulation of the
   full head (see `notes/RECHERCHE_PHASES_1-4.md`, section 5). Report as an estimate, never as a result.
4. The Fieldhouse 2026 ASIC number (2.0 nJ, 0.7 activity) is not comparable to our VCD numbers.

## 5. What is solid enough for a paper today

- Methodology: end-to-end, open, reproducible cost accounting of logic-gate networks on SKY130:
  training → netlist → synthesis → placement → gate-level simulation on real data → VCD power.
- Quantitative periphery breakdown on ASIC (H1–H3) with explicit lower-bound caveats.
- Negative results with fair baselines: L5, L6, L7, H4, L2/L3 (in their own cost model).

Prerequisites before submission:
- ≥3 seeds everywhere;
- SOTA-level training recipe or released models;
- wire-aware power (routed parasitics, SPEF) and a timing-annotated simulation for glitches;
- per-word clock gating in the input buffer.
