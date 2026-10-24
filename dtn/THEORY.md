# Dominance-Tree Networks — theory

Inventor: Badreddine Benfettouma. Date of this note: 2026-10-21.

## 1. Setting

A quantized ReLU layer computes, for an input `x ∈ {0..255}^N` (non-negative: pixels or outputs
of a previous ReLU layer),

    acc_i = w_i · x + b_i,      out_i = f(acc_i),

with integer weights `w_i`, integer bias `b_i` and an output function `f` that is
**monotone non-decreasing** and **zero on (-inf, T0]**, for example the requantization
`f(a) = clamp(round(a·M / 2^SH), 0, 255)` (then `T0 = floor((2^(SH-1) - 1) / M)`).
In sparse networks most outputs are 0 for a given input (2–10 % active in our models).
The goal is to compute **exactly** the same outputs while reading as few weights as possible
from the weight SRAM, whose reads dominate the energy of a near-memory engine.

## 2. Dominance trees

**Definition.** A dominance forest on the neurons of a layer is a set of rooted trees such that
for every edge (parent `p`, child `c`):

    w_c = w_p − D_c   with   D_c ≥ 0 element-wise,      b_c = b_p − d_c   with   d_c ≥ 0.

**Lemma 1 (dominance).** For every `x ≥ 0`: `acc_c ≤ acc_p`, hence `out_c ≤ out_p`.

*Proof.* `acc_p − acc_c = D_c · x + d_c ≥ 0` because every term is a product of non-negative
numbers; `f` is monotone. ∎

**Theorem 1 (exact skipping).** If `out_p = 0` then `out_d = 0` for every descendant `d` of `p`.
An evaluation that visits the forest top-down and skips the subtree of every inactive node
returns exactly the dense outputs.

*Proof.* By Lemma 1 along the path from `p` to `d`: `out_d ≤ out_p = 0` and `out_d ≥ 0`. ∎

**Corollary (incremental evaluation).** `acc_c = acc_p − D_c·x − d_c`: a visited child costs
one dot product with `D_c ≥ 0`, which is small (few bits, Section 5), not with `w_c`.

**Corollary (monotone early stop for children).** Reading the bit-planes of `D_c` from the most
significant one, every partial result is an upper bound of `acc_c` that only decreases; the
child is certified inactive as soon as the partial result is `≤ T0`. No error term is needed.

## 3. Expressivity

**Proposition 1.** Any dense ReLU layer with `M` neurons is exactly a set of leaves of a dominance
forest with at most `⌈M/C⌉` additional root neurons ("gates") of fan-out `C`.

*Proof.* Group the neurons in blocks of `C`; for a block `B` set `w_g = max_{i∈B} w_i`
(element-wise) and `b_g = max_{i∈B} b_i`; then `D_i = w_g − w_i ≥ 0` and `d_i = b_g − b_i ≥ 0`. ∎

So dominance is not a restriction of what a layer can compute, only of how cheaply it can be
skipped; training with the constraint (Section 6) makes the gates *also* useful features and
makes the forest *tight* (inactive gates for most inputs).

## 4. Cost model and scaling laws

Let `q` be the probability that a leaf neuron is active, `c_r` the cost (SRAM words) of
evaluating a root and `c_c` that of a visited child (`c_c ≤ c_r` because `D` has few bits).

**Depth 2.** `M = R(1 + C)` neurons, `R` roots with `C` children each. A root must be active
whenever one of its children is (Lemma 1), and in a tight forest it is active about only then:
`p_root ≈ 1 − (1 − q)^C ≈ C·q` for `C·q ≪ 1`. The expected cost per layer is

    W(C) = R·c_r + R·p_root·C·c_c ≈ M·c_r/(1 + C) + M·q·c_c·C.

Minimising over `C`:  `(1 + C*)^2 = c_r / (q·c_c)`, i.e. **`C* ≈ sqrt(c_r / (q·c_c))`** and

    W* ≈ 2·M·sqrt(q · c_r · c_c)       versus      W_dense ≈ M·c_r

(an exact dense evaluation must at least certify every neuron). The gain
`W_dense / W* ≈ ½·sqrt(c_r / (q·c_c))` grows as `1/sqrt(q)`: with `q = 2 %` and `c_c ≈ c_r`
it is ≈ 3.5, the order of magnitude measured on MNIST and digits (Section 7).

**Depth d.** With branching `b` at every level, the number of visited nodes is about `b` times
the number of active nodes over all levels; with the same tightness assumption the active
probability at level `ℓ` is `≈ q·b^(d−ℓ)`, giving

    W_d ≈ d · M · q^(1 − 1/d) · c        (b = q^(−1/d)),

minimised for `d* = ln(1/q)`:

    W_min ≈ e · M · q · ln(1/q) · c.

**Interpretation.** Exact computation becomes proportional to the number of *active* neurons
`M·q`, with only a logarithmic overhead — the same law as searching `M·q` items in a balanced
search tree. A dense layer, even with perfect certified early termination, stays `Ω(M)`.

The assumption to check experimentally is *tightness*: `P(parent active) ≈ P(some child active)`.
Training controls it (Section 6); the measured ratio is reported in Section 7.

## 5. Delta coding of children

`D_c ≥ 0` is stored with `NB` bits (`NB` = bit length of the largest gap of the layer). With the
softplus parametrisation, gaps concentrate near small values; `NB` = 4–6 in our models instead
of 8 bits for a weight, i.e. a lossless compression of the children of 25–50 %, on top of skipping.

## 6. Training

Parametrise `w_c = w_p − α·softplus(θ_c)`, `b_c = b_p − α·softplus(β_c)` (`α` ≈ the weight scale).
The constraint holds for every value of the free parameters, so plain SGD/Adam applies.
Quantization preserves it exactly: roots are rounded to int8, gaps are rounded to non-negative
integers, and the child's integer weights are *defined* as `w_p − D_c`.
An `L1` penalty on the hidden activations sets the density `q`; with the dominance constraint,
the network learns hierarchies of detectors (a child fires on a subset of its parent's inputs).

## 7. Measurements (to be completed with the multi-seed sweep)

Same exact skipping rules for every model (zero-word skipping, MSB-first certified stop for
ReLU-zero, certified argmax), 32-bit words of a bit-plane layout:

| Dataset | Model | Accuracy | Words / inference |
|---|---|---|---|
| MNIST (8 epochs, 1 seed) | dense 256-128, L1 0.01 | 96.56 % | 18 837 |
| MNIST | DTN 64×(1+3) – 32×(1+3), L1 0.003 | 96.86 % | 8 967 |
| MNIST | dense 256-128, L1 0.03 | 94.06 % | 11 602 |
| MNIST | DTN 32×(1+7) – 16×(1+7), L1 0.01 | 95.32 % | 4 947 |
| digits (3 seeds) | dense 120-60, L1 0.03 | 97.78 % | 1 826 |
| digits (3 seeds) | DTN 40×(1+4) – 20×(1+4), L1 0.01 | 97.26 % | 930 |

Dominance violations (a skipped child that would have been active): 0 in every run.

## 8. Hardware

See SPEC.md. The engine reads, per root, a bias word and its sign-magnitude bit-planes
(MSB-first, certified stop), and per child of an active root a gap word and its gap planes
(MSB-first, monotone stop). Words whose 32 activations are all zero are never read.
Exactness is proven formally on a reduced instance (formal/dtn).
