# Certified Early Termination for bit-serial MVM: theory

This note covers the mathematics behind **Certified Early Termination (CET)** for the near-memory
matrix-vector engine planned next to the RV32I core. It gives definitions, proves soundness and
tightness, adds cheaper bounds, proves the decision rules exact, builds an expected-cost model,
and lists the invariants a formal tool must prove. `model/theory_check.py` checks every lemma,
proposition and theorem, either exhaustively on tiny instances or by random testing. Corollaries
3.1 and 5.4 follow directly from checked statements. The results are in section 8.

**Verdict, in short.** The mathematics holds. The popcount interval is sound for every digit
code. It is exactly tight for two's complement and free signed digits, tight up to a permutation
of the activations for sign-magnitude-like codes, and only sound for NAF. The decision rules (ReLU
zero, any monotone requantization, argmax) give bit-exact results. The cost model predicts the
measurements to within 1 to 2 % (section 6.6). The same model also shows the limit of the popcount
idea. Once the SRAM words for metadata are counted, popcounts **barely beat the metadata-free box
baseline B1, and on dense int8 weights they lose**: savings appear only when the least
significant weight planes are sparse (pruned or power-of-two weights) or the activations are
dense. The reason is structural (prediction P3, section 6.5). Read the evaluation plan with that
in mind.

---

## 1. Setting and definitions

### 1.1 The computation

One output row computes

$$y = b + \sum_{j=1}^{N} w_j x_j ,\qquad w_j \in \mathbb{Z}\ (\text{int8, or int}B),\quad x_j \in \{0,\dots,2^{X}-1\}\ (\text{uint8 after ReLU}),$$

where $b$ is an optional integer bias that seeds the accumulator. The matrix has $R$ rows that
share the same activation vector $x$. Activations are **non-negative**, and every bound below
relies on that.

$S_x = \sum_j x_j$ is the activation sum.

### 1.2 Order statistics

**Definition 1.1.** Let $x_{(1)} \ge x_{(2)} \ge \dots \ge x_{(N)}$ be the entries of $x$ in
decreasing order, and $x_{[1]} \le \dots \le x_{[N]}$ the same entries in increasing order. For
$0 \le m \le N$:

$$\mathrm{Top}(m) = \sum_{i=1}^{m} x_{(i)}, \qquad \mathrm{Bot}(m) = \sum_{i=1}^{m} x_{[i]} = S_x - \mathrm{Top}(N-m).$$

The table $\mathrm{Top}(0..N)$ is computed **once per input vector** and shared by all $R$ rows.
$\mathrm{Bot}$ needs no second table. For a column subset $C$, $\mathrm{Top}_C$ and
$\mathrm{Bot}_C$ are the same quantities for $x$ restricted to $C$.

**Lemma 1.1 (subset sums).** For every $S \subseteq \{1..N\}$ with $|S| = m$:
$\mathrm{Bot}(m) \le \sum_{j\in S} x_j \le \mathrm{Top}(m)$, and both ends are attained.

*Proof.* The $i$-th largest value of the sub-multiset $\{x_j : j \in S\}$ is at most $x_{(i)}$.
Summing over $i \le m$ gives the upper bound, and the lower bound is symmetric. The indices of the
$m$ largest (or smallest) entries attain the bounds. ∎

**Lemma 1.2 (shape).** $\mathrm{Top}(0)=0$ and $\mathrm{Top}(N)=S_x$. The increment is
$\mathrm{Top}(m)-\mathrm{Top}(m-1) = x_{(m)} \ge 0$ and does not increase with $m$, so Top is
non-decreasing and (discretely) concave. Bot is non-decreasing and convex, and
$\mathrm{Bot}(m) \le \mathrm{Top}(m)$. Both properties need $x \ge 0$.

**Lemma 1.3 (disjoint pairs).** Let $p+q \le N$. Over disjoint sets $P, Q$ with $|P| = p$ and
$|Q| = q$:

$$\max \Big(\sum_P x - \sum_Q x\Big) = \mathrm{Top}(p) - \mathrm{Bot}(q), \qquad \min\Big(\sum_P x - \sum_Q x\Big) = \mathrm{Bot}(p) - \mathrm{Top}(q).$$

*Proof.* Lemma 1.1 applied to each sum gives the inequalities. Fix one sorting permutation $\pi$
(ties broken by index). Then $P = \pi(1..p)$ and $Q = \pi(N-q+1..N)$ are disjoint because
$p+q \le N$, and they attain the maximum. The minimum is symmetric. ∎

### 1.3 Digit codes and bit-planes

**Definition 1.2 (digit code).** A $P$-plane digit code is defined by:

- coefficients $c_0, \dots, c_{P-1}$, listed in **read order** (most significant first);
- for each column, a set $\mathcal{C}$ of valid code words $d = (d_0, \dots, d_{P-1})$ with
  $d_t \in \{-1, 0, 1\}$, each encoding $w = \sum_t c_t d_t$.

A row's **plane** $t$ is the vector $d_t = (d_{t,1}, \dots, d_{t,N})$. In SRAM it is one 32-lane
word per 32 columns, or two bit-vectors (`+1` mask and `-1` mask) for signed digits. The plane's
**split popcounts** are $m_t^+ = \#\{j : d_{t,j} = +1\}$ and $m_t^- = \#\{j : d_{t,j} = -1\}$. The
**plane sum** is

$$\pi_t = \sum_j d_{t,j} x_j = P_t^+ - P_t^-, \qquad P_t^\pm = \sum_{j:\, d_{t,j} = \pm 1} x_j ,$$

so $y = b + \sum_t c_t \pi_t$.

The codes studied here ($B$ = weight width):

| code | planes, coefficients (read order) | digit sets | range |
|---|---|---|---|
| **2C** two's complement | $t=0$: sign bit, $c_0 = -2^{B-1}$; $t \ge 1$: $c_t = 2^{B-1-t}$ | $\{0,1\}$; every pattern valid | $[-2^{B-1}, 2^{B-1}-1]$ |
| **SM** sign-magnitude | $t=0$: sign word $s$, $c_0 = 0$; $t \ge 1$: $c_t = 2^{B-1-t}$, $d_{t,j} = (1-2s_j)\,a_{t,j}$ | $\{0, \sigma_j\}$, where $\sigma_j = 1-2s_j$ | $[-(2^{B-1}-1), 2^{B-1}-1]$ |
| **SF** sign-folded 2C | stored exactly as 2C; plane $t \ge 1$ is read XOR the sign word: $c_0 = -1$, $d_{0,j} = s_j$, $d_{t,j} = (1-2s_j)(b_{t,j} \oplus s_j)$ | $\{0, \sigma_j\}$ | full 2C range |
| **SD** free signed digit | $c_t = 2^{P-1-t}$ | $\{-1,0,1\}$; every vector valid (redundant) | $[-(2^P-1), 2^P-1]$ |
| **NAF** non-adjacent form | $c_t = 2^{P-1-t}$, $d_{t,j}\, d_{t+1,j} = 0$ | $\{-1,0,1\}$; one code word per value | |

**Lemma 1.4 (sign folding).** If $b$ is the $B$-bit 2C pattern of $w$ with sign bit $s$, then
$w = -s + \sum_{t \ge 1} 2^{B-1-t} (1-2s)(b_t \oplus s)$.

*Proof.* For $s = 0$ the identity is plain 2C. For $s = 1$:
$w = -2^{B-1} + \sum 2^{B-1-t} b_t = -2^{B-1} + (2^{B-1}-1) - \sum 2^{B-1-t}(1-b_t) = -1 - \sum 2^{B-1-t}(b_t \oplus 1)$. ∎

So SF is the one's-complement magnitude $|w| - s$ under a sign. A negative weight close to zero
has sparse high planes, as in SM, yet $-2^{B-1}$ stays representable and the stored bits are
ordinary 2C. Reading SF costs one XOR per lane. SM cannot hold $-2^{B-1}$, so it needs symmetric
quantization (or saturating $-128$ to $-127$ offline). The reference model must use the same
weights.

**Definition 1.3 (product and sign-coupled codes).** After a read prefix $d_0..d_{k-1}$, column $j$
may take any completion in a set of the form $\prod_{t \ge k} E_t(j)$ with $0 \in E_t(j)$:

- the code is **column-uniform product** if $E_t(j)$ is the full alphabet for every column (2C:
  $\{0,1\}$; SD: $\{-1,0,1\}$);
- it is **sign-coupled** if $E_t(j) = \{0, \sigma_j\}$ depends on the column's sign, which is read
  first (SM, SF);
- NAF is **not** a product code, because the non-adjacency constraint couples planes $t$ and $t+1$.

### 1.4 MSB-first state

After $k$ planes have been read:

$$A_k = b + \sum_{t<k} c_t \pi_t \ \text{(known)}, \qquad R_k = \sum_{t \ge k} c_t \pi_t \ \text{(unread)}, \qquad y = A_k + R_k .$$

The **information state** $\mathcal{I}_k$ is $x$, the read planes, and the stored metadata
$(m_t^\pm)_{t \ge k}$. The **feasible completion set** $F_k$ holds every completion
$(d_t)_{t \ge k}$ that is valid for the code, extends the read prefix, and has the stored
popcounts. $F_k$ is never empty, because it contains the true row.

---

## 2. Soundness

**Theorem 1 (soundness).** For every plane $t$ define:

$$u_t = \mathrm{Top}(m_t^+) - \mathrm{Bot}(m_t^-), \qquad l_t = \mathrm{Bot}(m_t^+) - \mathrm{Top}(m_t^-),$$

$$h_t = \begin{cases} c_t u_t & c_t \ge 0 \\ c_t l_t & c_t < 0 \end{cases}, \qquad \ell_t = \begin{cases} c_t l_t & c_t \ge 0 \\ c_t u_t & c_t < 0 \end{cases}, \qquad \mathrm{Hi}_k = \sum_{t \ge k} h_t, \quad \mathrm{Lo}_k = \sum_{t \ge k} \ell_t .$$

Then $\mathrm{Lo}_k \le R_k \le \mathrm{Hi}_k$ for **every** digit code with digits in
$\{-1,0,1\}$, every row, every $x \ge 0$ and every $k$.

For 2C after the sign plane, every unread coefficient is $2^b > 0$ and $m^- = 0$, so the interval
reduces to the formula in the brief:
$\big[\sum_b 2^b \mathrm{Bot}(m_b),\ \sum_b 2^b \mathrm{Top}(m_b)\big]$.

*Proof.* $P_t^+$ and $P_t^-$ are sums over disjoint index sets of sizes $m_t^+$ and $m_t^-$.
Lemma 1.1 bounds each one, so $l_t \le \pi_t \le u_t$. Multiplying by $c_t$ (which flips the
interval when $c_t < 0$) gives $\ell_t \le c_t \pi_t \le h_t$. Summing over the unread planes
completes the proof. No property of the code beyond the digit alphabet is used, so the theorem
covers NAF and other canonical recodings. ∎

**Corollary 2.1 (enclosure).** $L_k = A_k + \mathrm{Lo}_k \le y \le U_k = A_k + \mathrm{Hi}_k$,
and $L_P = U_P = y$.

**Corollary 2.2 (nesting).** $L_k \le L_{k+1}$ and $U_{k+1} \le U_k$. This holds for **any**
per-plane enclosure $\ell_t \le c_t \pi_t \le h_t$, so it also holds for every cheaper bound in
section 4. The width $W_k = U_k - L_k = \sum_{t \ge k} (h_t - \ell_t)$ does not increase with $k$.

*Proof.* $U_k - U_{k+1} = h_k - c_k \pi_k \ge 0$, and the lower bound is symmetric. ∎

**Proposition 2.3 (the B1 box baseline).** Without metadata, after the sign plane, each plane
satisfies $\pi_t \in [-X^-, X^+]$, where $X^\pm$ is the activation sum over the columns that may
carry a $\pm 1$ digit:

- 2C: $X^+ = S_x$ and $X^- = 0$;
- SM and SF: $X^\pm = \sum_{j \in S^\pm} x_j$, where $S^+$ ($S^-$) are the positive (negative)
  columns;
- SD and NAF: $X^\pm = S_x$.

This is baseline **B1**. For 2C, Theorem 1 always lies inside it, because $\mathrm{Top}(m) \le S_x$.
For sign-coupled codes it **need not**: the global $\mathrm{Top}(m^+)$ may use columns outside
$S^+$ and exceed $X^+$. Measured on a dense int8 layer, the plain global bound is then *worse*
than B1 (scheme `popg` in section 6.6). Proposition 2.4 fixes this.

**Proposition 2.4 (sign-aware complement refinement).** Suppose a plane's $+1$ digits are confined
to a column set $C^+$ of size $n^+$ and activation sum $X^+$, both known once the sign word has
been read. Then

$$\max\!\big(\mathrm{Bot}(m^+),\ X^+ - \mathrm{Top}(n^+ - m^+)\big) \ \le\ P^+ \ \le\ \min\!\big(\mathrm{Top}(m^+),\ X^+ - \mathrm{Bot}(n^+ - m^+)\big),$$

and the same holds for $P^-$ with $C^-$. The refined range lies inside both Lemma 1.1's range and
the box $[0, X^+]$.

*Proof.* $P^+ = X^+ - \sum_{C^+ \setminus \mathrm{supp}} x$, where the removed set has
$n^+ - m^+$ elements. Lemma 1.1 bounds its sum globally, because restricted order statistics are
dominated by global ones. ∎

The refinement costs almost nothing in hardware: $n^\pm$ is the popcount of the sign word, and
$X^- = P_0$ is the sign plane's own sum ($A_1 = -X^-$ in SF). For 2C it changes nothing, since
$S_x - \mathrm{Bot}(N-m) = \mathrm{Top}(m)$. The same argument applies to NAF: a non-zero digit
at plane $t$ forces a zero at $t+1$ in that column, so plane $t+1$ is confined to the complement of
$\mathrm{supp}(d_t)$, whose activation sum is known after plane $t$ is read. This NAF case is a
direct corollary of Proposition 2.4 but is not exercised by the script.

**Proposition 2.5 (box invariance).** With B1 boxes, 2C and SF storage produce **identical**
intervals $[L_k, U_k]$ for every $k \ge 1$. SM produces different ones.

*Proof sketch.* By Lemma 1.4, $2^b(P_b^{+,SF} - P_b^{-,SF}) = 2^b(P_b^{2C} - X^-)$ and the SF box
is $2^b(S_x - X^-)$. Summing over read and unread planes, the $X^-$ terms add up to the 2C sign
term $-2^{B-1} X^-$. ∎ (Checked exhaustively.)

---

## 3. Tightness

**Theorem 2 (tightness).** Fix $x$, a row, and $k$.

**(a) Column-uniform product codes (2C for every $k \ge 0$; free SD).**
$\max_{F_k} R_k = \mathrm{Hi}_k$ and $\min_{F_k} R_k = \mathrm{Lo}_k$. The interval is the exact
range of the unread contribution over all weight bits that share the stored popcounts.

*Proof.* $F_k$ is a product over planes, so each plane can be optimized independently. For
$c_t \ge 0$, place the $+1$ digits on the $m_t^+$ largest activations and the $-1$ digits on the
$m_t^-$ smallest. These sets are disjoint because $m_t^+ + m_t^- \le N$ (Lemma 1.3), and the
placement is valid because every column admits the full alphabet. For $c_t < 0$, reverse the
placement. Each plane then reaches $h_t$. ∎

Only the end points are guaranteed: the attainable values between them are generally a strict
subset of the integers in between. That is harmless, because every decision rule below looks only
at the end points.

**(b) Sign-coupled codes (SM, SF), $k \ge 1$.** Given the signs, $F_k$ is the product over
planes of $\{P^+ \subseteq S^+,\ |P^+| = m^+\} \times \{P^- \subseteq S^-,\ |P^-| = m^-\}$:
magnitude bits are free, by Lemma 1.4 for SF. Hence the **exact** range uses restricted
statistics:

$$\max_{F_k} R_k = \mathrm{Hi}^*_k := \sum_{t \ge k,\ c_t>0} c_t\big(\mathrm{Top}_{S^+}(m_t^+) - \mathrm{Bot}_{S^-}(m_t^-)\big) \ \ (+\text{ the symmetric terms for } c_t<0).$$

The global bound satisfies $\mathrm{Hi}_k \ge \mathrm{Hi}^*_k$, with equality iff every unread plane
has $\mathrm{Top}_{S^+}(m_t^+) = \mathrm{Top}(m_t^+)$ and
$\mathrm{Bot}_{S^-}(m_t^-) = \mathrm{Bot}(m_t^-)$. The first equality holds **iff** $S^+$ contains
every column with $x_j > x_{(m)}$ plus at least $m - \#\{j : x_j > x_{(m)}\}$ columns with
$x_j = x_{(m)}$ (where $m = m_t^+$). For a post-ReLU vector with $z$ zeros and $m \le z$, the
second reduces to "$S^-$ contains at least $m$ zero activations".

*Proof.* The $i$-th largest value of $x|_{S^+}$ is at most $x_{(i)}$, and the sums are equal iff
the values are equal term by term for $i \le m$. That is exactly the stated condition. ∎

**(c) Permutation tightness (SM, SF, $k \ge 1$).** The global interval is exact up to a
rearrangement of the activations:

$$\mathrm{Hi}_k = \max_{x' \in \mathrm{Perm}(x)}\ \max_{F_k} R_k(x'), \qquad \mathrm{Lo}_k = \min_{x'}\ \min_{F_k} R_k(x').$$

So $\mathrm{Hi}_k$ is the **smallest valid bound among all bounds that depend on $x$ only through
its sorted values**, which is exactly the class computable from one table shared by all rows.

*Proof.* Let $M^\pm = \max_{t \ge k} m_t^\pm$. Then $M^+ \le |S^+|$ and $M^- \le |S^-|$, so
$M^+ + M^- \le N$. Place the $M^+$ largest values on columns of $S^+$ and the $M^-$ smallest on
columns of $S^-$. Restricted and global statistics then agree for every unread plane, and (b)
applies. ∎

Proposition 2.4 uses extra row information ($n^\pm$, $X^\pm$), so (c) does not limit it; it can
only be tighter.

**(d) Canonical recodings (NAF, Booth).** Only soundness holds. A counterexample has $N = 2$,
$x = (10, 1)$, and two adjacent NAF planes of weight 2 and 1, each with one $+1$ digit. The bound
is $2 \cdot 10 + 1 \cdot 10 = 30$, but non-adjacency forces different columns, so the true maximum
is $2 \cdot 10 + 1 = 21$. The exhaustive search finds the bound strictly loose in 29 to 47 % of
(row, $k$, $x$) cases.

**Corollary 3.1 (information-optimal stopping).** For column-uniform product codes (2C, SD) with
exact Top and Bot, suppose CET has not stopped at step $k$. Then **no** stopping rule that sees
only $\mathcal{I}_k$ (activations, read planes, popcounts) can stop at $k$ and still be correct for
every consistent weight matrix.

*Proof.*
- **Monotone $f$:** $f(L_k) \ne f(U_k)$, and both end points are attained by consistent rows
  (Theorem 2a), so any output is wrong for one of them.
- **Argmax:** the rows' completions are independent, so the corner where $y_i = L_i$ and
  $y_r = U_r$ for all $r \ne i$ is attained jointly. If no row is certified (section 5), then for
  every candidate $i$ that corner makes some other row win.

For SM and SF the statement holds with restricted statistics. With global statistics it holds in
the permutation sense of (c). ∎

**Consequence.** For 2C, the popcount certificate is the best possible use of popcount metadata.
To stop earlier you would need *more informative metadata*, not a smarter rule.

---

## 4. Cheaper sound bounds (majorants)

**Lemma 4.0 (monotone substitution).** Theorem 1, Corollary 2.2 and Proposition 2.4 stay valid when
Top is replaced by any $\widehat{T} \ge \mathrm{Top}$ and Bot by any $\widehat{B} \le \mathrm{Bot}$
(pointwise). The proof is the same.

**Proposition 4.1 (tangent lines of the concave Top).** For $1 \le m \le N$ and $0 \le n \le N$:

$$\mathrm{Top}(n) \le \mathrm{Top}(m) + (n-m)\,x_{(m)} ,$$

that is, $\mathrm{Top}(n) \le \mathrm{Top}(m) + (n-m)x_{(m)}$ for $n \ge m$ and
$\mathrm{Top}(n) \le \mathrm{Top}(m) - (m-n)x_{(m)}$ for $n < m$. The convex Bot has the mirror
property: $\mathrm{Bot}(n) \ge \mathrm{Bot}(m) + (n-m)\,x_{[m]}$.

*Proof.*
- For $n \ge m$: $\mathrm{Top}(n) - \mathrm{Top}(m) = \sum_{i=m+1}^{n} x_{(i)} \le (n-m)x_{(m+1)} \le (n-m)x_{(m)}$.
- For $n < m$: $\mathrm{Top}(m) - \mathrm{Top}(n) = \sum_{i=n+1}^{m} x_{(i)} \ge (m-n)x_{(m)}$.

The proof for Bot is symmetric. For $m = 0$, use slope $x_{(1)}$: $\mathrm{Top}(n) \le n\,x_{\max}$. ∎

**Proposition 4.2 (tangent envelope and its gap).** Let $M \subseteq \{0..N\}$ contain $0$ and
$N$, and let $g_m = x_{(\max(m,1))}$. Define

$$\widehat{T}_M(n) = \min_{m \in M}\big[\mathrm{Top}(m) + (n-m) g_m\big].$$

Then $\widehat{T}_M \ge \mathrm{Top}$, with equality on $M$, and

$$\max_n\big(\widehat{T}_M(n) - \mathrm{Top}(n)\big) \le \max_{a<b \text{ consecutive in } M} \frac{(b-a)(g_a - g_b)}{4}.$$

*Proof.* On $[a,b]$, $\widehat{T}_M \le \min(\ell_a, \ell_b)$ (the two tangents), and Top lies
above its chord by concavity. The chord's slope $s$ is an average of $x_{(a+1..b)}$, so
$g_b \le s \le g_a$. The triangle formed by the two tangents and the chord has height
$(b-a)(g_a - s)(s - g_b)/(g_a - g_b)$, which is at most $(b-a)(g_a - g_b)/4$ by AM-GM. ∎

Two examples:

- $M = \{0, N\}$ gives the brief's majorant $\min(n\,x_{\max}, S_x)$, with gap at most
  $N x_{\max}/4$.
- $M = \{0, 1, 2, 4, \dots, N\}$ needs only about $\log_2 N + 2$ ranks, found by selection rather
  than a full sort. The measured worst gap is 0.68 of the bound.

**Proposition 4.3 (rounding majorant; no sort needed).** If $\hat{x} \ge x$ elementwise, then
$\mathrm{Top}_{\hat{x}} \ge \mathrm{Top}_x$; if $\check{x} \le x$, then
$\mathrm{Bot}_{\check{x}} \le \mathrm{Bot}_x$. Round each activation up to $p+1$ significant bits:
$\hat{x} = \lceil x/2^e \rceil 2^e$ with $e = \max(0, \lfloor\log_2 x\rfloor - p)$. Then

$$\mathrm{Top}_x \le \mathrm{Top}_{\hat{x}} \le (1 + 2^{-p})\,\mathrm{Top}_x ,$$

and rounding down gives $(1 - 2^{-p})\mathrm{Bot}_x \le \mathrm{Bot}_{\check{x}} \le \mathrm{Bot}_x$.

*Proof.* $0 \le \hat{x}_j - x_j < 2^e \le x_j 2^{-p}$. Take the top-$m$ set of $\hat{x}$ and apply
the bound term by term. ∎

$\hat{x}$ takes only about $2(X-p) + 2^{p+1}$ distinct values (17 for $X = 8$, $p = 1$), so
$\mathrm{Top}_{\hat{x}}(m)$ is a piecewise-linear function of a **histogram** of bucket counts.
Hardware needs a few counters and a priority scan, not a sorting network. The measured worst
excess is 0.98, 0.49, 0.24 and 0.12 for $p = 0..3$, against bounds of 1, 1/2, 1/4 and 1/8.

**Proposition 4.4 (metadata rounding).** Top and Bot are non-decreasing in $m$. Storing a code
$\hat{m}^+ \ge m^+$ for the Top terms, and $\check{m} \le m$ for the Bot terms, keeps the bounds
sound, at a cost of $\mathrm{Top}(\hat{m}) - \mathrm{Top}(m) \le (\hat{m} - m)\,x_{(m+1)}$. A
ReLU decision needs only the upper side, so a single rounded-up code per plane suffices (scheme
`pop+q4`, a 16-level geometric code book).

**Proposition 4.5 (partition refinement).** Partition the columns into chunks $C_1..C_q$ and store
per-chunk popcounts $m_c$. Then

$$\sum_c \mathrm{Top}_{C_c}(m_c) \le \mathrm{Top}\Big(\sum_c m_c\Big), \qquad \sum_c \mathrm{Bot}_{C_c}(m_c) \ge \mathrm{Bot}\Big(\sum_c m_c\Big).$$

Finer chunks give tighter bounds and need more metadata and more tables. One chunk per 32-lane
SRAM word is the natural choice.

*Proof.* The union of the per-chunk top sets is a set of size $\sum_c m_c$; apply Lemma 1.1. ∎

---

## 5. Decision rules

**Proposition 5.1 (monotone output maps).** If $f: \mathbb{Z} \to \mathbb{Z}$ is non-decreasing,
$L \le y \le U$ and $f(L) = f(U)$, then $f(y) = f(L)$, because $f(L) \le f(y) \le f(U)$. Three
cases follow:

1. **ReLU**, $f(y) = \max(y, 0)$: $f(L) = f(U)$ iff $U \le 0$ or $L = U$. So "output 0 when
   $U_k \le 0$" is exact, and it is the only early stop ReLU allows; a positive output needs its
   exact value.
2. **Fused ReLU + requantization**,
   $f(y) = \mathrm{clamp}\big(\lfloor (My + 2^{s-1}) / 2^s \rfloor + z,\ q_{\min}, q_{\max}\big)$
   with $M \ge 0$. This $f$ is monotone, so stop when $f(L_k) = f(U_k)$. That covers three
   situations:
   - the zero bucket, $U_k < \tau_0$, where $\tau_0 > 0$ is the least $y$ with $f(y) > q_{\min}$
     (strictly more stops than ReLU alone);
   - saturation, $L_k \ge \tau_{\max}$;
   - any interior cell, once $[L_k, U_k]$ fits inside it.

   The rule needs **monotonicity, not linearity**: floor, half-up and half-even rounding all
   qualify. The hardware $f$ must be the bit-identical integer function used by the reference.
3. **Stickiness.** By Corollary 2.2, a certificate at step $k$ stays valid at every $k' \ge k$ with
   the same value. A detector that fires a few cycles late, after extra plane reads are already in
   flight, still outputs the right value; only reads are wasted.

**Proposition 5.2 (argmax certificate).** Let $i^* = \min\{i : y_i = \max_r y_r\}$; this is the
`numpy.argmax` convention. Write $i \rhd r$ ("$i$ surely ranks before $r$") iff either $r < i$ and
$L_i > U_r$, or $r > i$ and $L_i \ge U_r$. If $i \rhd r$ for every $r \ne i$, then $i = i^*$. At
most one index can be certified.

*Proof.* For $r < i$: $y_i \ge L_i > U_r \ge y_r$. For $r > i$: $y_i \ge y_r$. So $y_i$ is maximal
and strictly larger than every earlier row's value. ∎

**Proposition 5.3 (elimination, schedule independence, termination).**

1. If $q \rhd r$ for some $q$, then $r \ne i^*$, and row $r$ may be dropped and never read again.
2. Take any schedule that refines only non-eliminated, not fully read rows (one plane at a time,
   in any order). It stops after at most $R(P-1)$ refinements beyond the sign planes, and it stops
   with a certificate for $i^*$.

*Proof of 2.* Suppose there is no certificate and nothing left to refine. Then $i^*$, which is
never eliminated, is fully read, so $L_{i^*} = y_{i^*}$.

- Every non-eliminated row $r$ is also fully read, so $U_r = y_r$, and $i^* \rhd r$ by the
  definition of $i^*$.
- Every eliminated row $r$ was eliminated by some $q$:
  - if $q > r$, then $y_{i^*} \ge y_q \ge L_q > U_r$;
  - if $q < r$, then $y_{i^*} \ge U_r$. When $r < i^*$, equality would give $y_q = y_{i^*}$ with
    $q < i^*$, which contradicts the minimality of $i^*$, so the inequality is strict.

In every case $i^* \rhd r$, so $i^*$ is certified, contradicting the assumption. ∎

The schedule changes only the cost. Best-first (refine the open row with the largest $U$) is the
greedy choice and is the one the script uses.

**Corollary 5.4 (box constancy).** A decision $\delta(y_1..y_R)$ may be issued exactly when
$\delta$ is constant on the box $\prod_r [L_r, U_r]$. For $\delta = \mathrm{argmax}$, box constancy
is equivalent to the certificate of Proposition 5.2: test the corner $y_i = L_i$, $y_r = U_r$. For
product codes this is also the earliest correct stop (Corollary 3.1).

---

## 6. Expected cost

### 6.1 Accounting

$w_p = \lceil N/32 \rceil$ is the number of SRAM words per plane, sign plane included. A row costs

$$C_{\text{row}} = K\,w_p + \mu/32 \ \text{words},$$

where $K \in \{1..P\}$ is the number of planes read (the sign plane is always read first) and $\mu$
is the metadata in bits per row, packed and amortized. The baselines are:

- **B0**, no early termination: $P\,w_p$ words.
- **B1**, the box of Proposition 2.3: $K_{\text{box}}\,w_p$ words, with no metadata.

Popcount metadata costs $\mu = (P-1)\lceil \log_2(N+1) \rceil$ bits for one side (2C, or `pop+`),
twice that for both sides of a signed code, and $(P-1) \cdot 4$ bits with 4-bit codes.

### 6.2 Deterministic stopping bound

**Theorem 3.** Let $f$ be monotone and let $[a(y), b(y)]$ be the constancy cell of $f$ that
contains $y$. Define the distance to the cell boundary $d_f(y) = \min(y - a, b - y)$. For ReLU,
$d = -y$ when $y \le 0$ and $d = 0$ when $y > 0$. Then

$$K \le \bar{K} := \min\{k \ge 1 : W_k \le d_f(y)\} \le P .$$

*Proof.* If $W_k \le d$, then $L_k \ge y - W_k \ge a$ and $U_k \le y + W_k \le b$, so
$f(L_k) = f(U_k)$. Also $W_P = 0$, so the set is never empty. ∎

For ReLU the real trigger is weaker: it needs only the upper slack $\mathrm{Hi}_k - R_k \le W_k$.
That is why measured $K$ is below $\bar{K}$.

**Corollary 6.1 (tail-sum identity).** $W_k$ does not increase with $k$, so for $1 \le k < P$,
$\bar{K} > k \iff W_k > d_f(y)$. Hence

$$\mathbb{E}[\bar{K}] = 1 + \sum_{k=1}^{P-1} \Pr\big[d_f(y) < W_k\big] \ \ \ge\ \mathbb{E}[K],$$

and for ReLU, $\Pr[d < W_k] = \Pr[y > -W_k]$. The identity holds jointly, even when the widths vary
from row to row. In words, the expected number of planes is the sum over planes of the probability
that the margin is smaller than the residual width.

### 6.3 Closed form

**Proposition 6.2 (exponential negative tail).** Model M1 makes two assumptions:

- **geometric widths:** $W_k = \omega(2^{P-k} - 1)$. This holds exactly for B1 on 2C and SF, with
  $\omega = S_x$;
- **exponential negative tail:** $\Pr[y \le -t] = q\,e^{-t/\beta}$ for $t \ge 0$, where
  $q = \Pr[y \le 0]$. The symmetric Laplace law is the case $q = 1/2$.

Then, with $\lambda = \beta / \omega$,

$$\mathbb{E}[\bar{K}] = P - q \sum_{j=1}^{P-1} e^{-(2^j - 1)/\lambda} .$$

For $1 \ll \lambda \ll 2^{P-1}$:

$$\sum_{j \ge 1} e^{-(2^j-1)/\lambda} = \log_2 \lambda - C + \epsilon(\lambda), \qquad C = \frac{\gamma}{\ln 2} + \frac{1}{2} = 1.33275\ldots,$$

where $\epsilon$ combines an $O(\log\lambda / \lambda)$ term with a periodic term of amplitude below
$2 \cdot 10^{-6}$. **On average, each negative row therefore saves about $\log_2(\beta/\omega) - 1.33$
planes, and the overall saving is about $q(\log_2(\beta/\omega) - 1.33)$ planes per row.**

*Proof.* Use Corollary 6.1 with $\Pr[y > -W_k] = 1 - q\,e^{-W_k/\beta}$ and substitute
$j = P - k$. For the asymptotics: the Mellin transform of $\sum_{j \ge 1} e^{-2^j u}$ is
$\Gamma(s)/(2^s - 1)$. Its double pole at $s = 0$, where
$\Gamma(s)/(2^s - 1) = 1/(s^2 \ln 2) - (\gamma/\ln 2 + 1/2)/s + O(1)$, gives
$\log_2(1/u) - \gamma/\ln 2 - 1/2$. The poles at $s = 2\pi i k/\ln 2$ give the periodic term, with
amplitude $|\Gamma(2\pi i/\ln 2)|/\ln 2 \approx 8 \cdot 10^{-7}$ per pole. The factor $e^{1/\lambda}$
contributes $O(\log \lambda/\lambda)$. ∎ (Monte-Carlo agrees with the closed form, and the numerical
$C$ matches to $10^{-5}$.)

**Proposition 6.3 (requantization cells).** Model M2 assumes that $y$ falls uniformly inside its
interior cell of width $\Delta$ (a "dither" assumption), so $\Pr[d < W] = \min(1, 2W/\Delta)$. Then
for interior rows

$$\mathbb{E}[\bar{K}] = 1 + \sum_{k=1}^{P-1}\min(1, 2W_k/\Delta) .$$

With geometric widths, $W_{k+1} \le W_k/2$, so the tail is a geometric series and
$\mathbb{E}[\bar{K}] < k_\Delta + 2$, where $k_\Delta = \min\{k \ge 1 : 2W_k < \Delta\}$. Interior
cells save planes only when $\Delta$ is several times larger than $\omega$.

**Proposition 6.4 (metadata break-even).** Let $N$ be a multiple of 32. A scheme with $\mu$ bits of
metadata per row beats a metadata-free scheme **iff** it saves more than $\mu/N$ planes per row:

$$\mathbb{E}[K_{\text{free}}] - \mathbb{E}[K_{\text{meta}}] > \mu/N .$$

For popcounts this threshold is 0.77 planes at $N = 64$ and 0.25 at $N = 256$; with 4-bit codes it
is 0.44 and 0.11.

### 6.5 What the model predicts

- **P1, logarithmic law.** Savings grow with $\log_2(\text{margin}/\text{width})$. Tightening the
  residual bound by a constant factor $\rho$ on every unread plane saves only $q \log_2(1/\rho)$
  planes per row (in the regime of Proposition 6.2). Halving the width is worth half a plane at $q = 1/2$.
- **P2, a ceiling set by the negative fraction.** For ReLU, positive rows always read every plane,
  so even an oracle that stops each negative row after the sign plane reaches only
  $\text{words}/\text{B0} \ge 1 - q(P-1)/P$, about 0.58 at $q = 0.48$. B1 reaches 0.83; that is,
  it captures about 40 % of the oracle's saving.
- **P3, popcounts act only through the planes still unread when the certificate fires.** Under B1
  the typical stop is at $k \approx 5..7$ of 8, so the planes that matter are the **1 to 3 least
  significant** ones. On uniform int8 grids those planes are dense (about half ones). On a
  post-ReLU vector, $m_t \gtrsim \mathrm{nnz}(x)$ then gives $\mathrm{Top}(m_t) \approx S_x$, so
  popcounts carry almost no information there and $\omega_{\text{pop}} \approx \omega_{\text{box}}$.
  The sparsity of the **high** planes in SM/SF, which motivated the idea, does not help: those
  planes are read before any certificate can fire. (Measured per-plane ratio of the CET upper bound
  to the B1 upper bound, SF, $t = 1..7$: 0.52 0.87 0.98 1.00 1.00 1.00 1.00.)
- **P4, where popcounts do help.** They help when the low planes are sparse (pruned weights,
  power-of-two or other low-bit-density quantizers) or when activations are dense (a first layer,
  or no ReLU), because then $\mathrm{Top}(m) < S_x$.
- **P5, metadata break-even.** Proposition 6.4 decides the outcome, and it favors large $N$.
- **P6, requantization.** 8-bit outputs gain nothing beyond the zero bucket, because $\Delta$ is far
  smaller than $\omega$. Outputs of 4 bits or fewer, and saturation, do gain.
- **P7, argmax.** Row $r$ is eliminated once $W_{r} + W_{i^*} < y_{i^*} - y_r$ (a sufficient
  condition). The cost is dominated by the close competitors, and wide gaps are cheap.

### 6.6 What the simulation shows

The figures come from `model/theory_check.py`, section C6. The setup is synthetic: $N = 64$
($w_p = 2$), 8 layers of 256 rows, int8 per-row max-scaled Laplace weights, and
$x = \mathrm{clip}(\mathrm{round}(\max(z,0) \cdot 255/3))$ with $z \sim \mathcal{N}(0,1)$.
**Words** means SRAM words read per output, metadata included, normalized to B0 (16 words).
¹ marks a value from the break-even run, which uses other seeds; that run gives 0.880 for 2C B1
in scenario D.

| scenario (negative rows) | B1 box | best popcount scheme | vs B1 |
|---|---|---|---|
| A Laplace int8, ReLU $x$ (48 %) | **0.826** | 2C/SF `pop+q4` 0.880 | 1.065 (worse) |
| B same, bias for 70 % negative | **0.718** | 2C `pop+q4` 0.772 | 1.075 (worse) |
| C 75 %-pruned, ReLU $x$ (50 %) | 0.867 | SF `pop+q4` **0.848** | 0.978 |
| D power-of-two weights, ReLU $x$ (50 %) | 0.874 (SM: **0.828**¹) | 2C `pop+q4` 0.846 | 0.968 |
| E Laplace int8, dense $x$ (52 %) | **0.841** | 2C `pop+q4` 0.864 | 1.027 (worse) |
| F power-of-two weights, dense $x$ (49 %) | 0.903 | 2C `pop+q4` **0.820** | 0.908 |

Other measurements:

- **Planes read, scenario A:** B1 reads $K = 6.605$ planes; exact 2C popcounts read 6.581. The gain
  of 0.024 planes falls far short of the 0.77 break-even (P3, P5).
- **Model check:** Proposition 6.2, with a fitted exponential tail and $\omega = S_x$, predicts
  $\mathbb{E}[\bar{K}] = 7.050$ against a measured 6.991 (scenario A), and 6.475 against 6.346
  (scenario B). Across A to F the prediction is within 1 to 2 %.
- **Requantization, scenario A, B1:** 8-bit outputs read $K = 6.592$, which is the same as ReLU.
  4-bit outputs read 6.244, so 0.36 planes are saved, almost all of it through the wider zero
  bucket (P6).
- **Argmax, 10 rows, $N = 64$:** B1 reads 86.9 words against 160 for B0 (0.54). Popcounts read
  101.6 (2C) and 117.0 (SF), because metadata dominates.
- **Break-even at $N = 256$:** savings over B1 grow to 0.4 to 1.1 planes for scenarios C, D and F,
  well above the 0.11 threshold for 4-bit codes. The best cases are 2C in F (0.844 against 0.950
  for B1, 11 % fewer words) and SM in D (0.853 against 0.884, 3.5 % fewer). Scenario A is still
  worse than B1 (0.900 against 0.886).
- **SM against 2C/SF:** for power-of-two weights SM is clearly better even under B1 (0.828 against
  0.880 at $N = 64$). In SF, $-2^e$ maps to $2^e - 1$, which has **dense low planes**.

---

## 7. Hardware invariants for formal proof (SymbiYosys)

### 7.1 Reference engine and harness

This section describes the **reference engine** whose interface the RTL should keep, so that the
properties below apply directly:

- **Parameters:** `N` lanes, `XB` activation bits, `P` planes (sign plane included),
  `MB = clog2(N+1)`, accumulator width `AW`.
- **Per vector:** registers `x[N]` and the table `T[0..N]`, with `Bot(m) = T[N] - T[N-m]`.
- **Per row:**
  - inputs: `sgn` (SF or SM; plane 0 for 2C) and metadata `mp[t]`, `mn[t]` for `t = 1..P-1`;
  - state: `k` and `acc` ($A_k$), `hi_rem` ($\mathrm{Hi}_k$), `lo_rem` ($\mathrm{Lo}_k$);
  - outputs: `done`, `y_out` / `q_out` / `idx_out`.
- **Memory:** the same contract as `formal/wrapper.sv`: synchronous read, data valid one cycle
  after an enabled request.

The harness follows the repository's existing flow: an SV wrapper read with
`read_verilog -formal`, around the VHDL read through the GHDL plugin. It uses free constants:

```systemverilog
(* anyconst *) reg [N*XB-1:0]  f_x;     // the activation vector
(* anyconst *) reg [R*P*N-1:0] f_w;     // every stored bit-plane of every row (read order)
(* anyconst *) reg [N-1:0]     f_mask;  // an arbitrary column subset (table lemma)
```

Reference functions are formal-only, written as plain SV functions with constant loops:
`digit(r,t,j)` per code (2C, SF, SM), `plane_sum(r,t)`, `ref_partial(r,k)` $= b + \sum_{t<k} c_t \pi_t$,
`ref_y(r) = ref_partial(r,P)`, and `ref_hi(r,k)` / `ref_lo(r,k)`, which evaluate the engine's
bound formula from `popcount` of `f_w` and the table.

### 7.2 Properties

**Assumptions (environment):**

| id | property |
|---|---|
| A1 | The memory returns `f_w[(r*P+t)*N +: N]` one cycle after a request for plane `t` of row `r`. |
| A2 | The metadata returns popcounts of the stored planes. For 2C, `mp[t] == popcount(plane_t)` and `mn[t] == 0`. For SM and SF, `mp[t] == popcount(v_t & ~sgn)` and `mn[t] == popcount(v_t & sgn)`, with `v_t = plane_t ^ sgn` for SF and `v_t = plane_t` for SM. With quantized metadata, use `>=` / `<=` in the sound direction (Proposition 4.4). |
| A3 | `x` equals `f_x` once loaded and is stable for the whole job; reset and start follow the protocol. |

A2 sits outside the proof boundary. Either generate popcounts in hardware on the write path (and
then prove `meta == popcount(plane)` as an invariant over SRAM writes), or validate the offline
generator exhaustively in Python.

**Invariants** (assert; also the strengthening lemmas for k-induction):

| id | property | meaning |
|---|---|---|
| I1 | `m == popcount(f_mask) -> T[N]-T[N-m] <= sum(f_mask & x) <= T[m]`; `T[0]==0`; `T[N]==sum(x)` | table soundness (Lemma 1.1); `cover` that `sum == T[m]` is reachable (tightness) |
| I2 | `busy -> acc == ref_partial(r,k)` | partial-sum exactness |
| I3 | `busy -> hi_rem == ref_hi(r,k) && lo_rem == ref_lo(r,k)` | bound bookkeeping (initialize, then subtract $h_t$, $\ell_t$ on each plane) |
| I4 | `busy -> acc+lo_rem <= ref_y(r) && ref_y(r) <= acc+hi_rem` | **Theorem 1 in hardware** |
| I5 | plane consumed `-> acc+hi_rem <= $past(acc+hi_rem) && acc+lo_rem >= $past(acc+lo_rem)` | nesting (Corollary 2.2) |
| I6 | `busy && k==P -> hi_rem==0 && lo_rem==0` | exact after the last plane |
| I7 | every `AW`-bit signed quantity equals its recomputation at `AW+2` bits | no overflow |
| I8 | plane requests are issued in read order, each at most once per job; `k` increments only on returned data | MSB-first, read-once |

**Decision properties** (the end-to-end statements):

| id | property |
|---|---|
| D1 | `done && mode==RELU -> y_out == (ref_y > 0 ? ref_y : 0)` |
| D2 | `done && mode==RQ -> q_out == f_ref(ref_y)`, where `f_ref` is an independent SV copy of the integer requantization (Proposition 5.1) |
| D3 | `done && mode==ARGMAX -> forall r: (r<idx -> ref_y[idx] > ref_y[r]) && (r>idx -> ref_y[idx] >= ref_y[r])` |
| D3e | `elim[r] -> exists q: q ⊳ r` on the stored $(L,U)$, and `elim[r] -> r != idx_ref` (Proposition 5.3) |
| D4 | miter: two instances (`et_en=1`, `et_en=0`) on the same free constants; both `done -> outputs equal` ("early-terminated output == full output") |

**Cost and progress:**

| id | property |
|---|---|
| C1 | `words_read <= meta_words + P*w_p` per row, and never more than the B0 instance plus metadata |
| C2 | bounded liveness as safety: `start -> done` within `P*(w_p+lat)+c` cycles (counter assertion) |
| C3 | stickiness (debug mode that continues after a certificate): the certified value is unchanged at every later `k` |

The core assertions, in the immediate-assertion style that open-source Yosys accepts:

```systemverilog
always @(*) if (f_valid && busy) begin
    assert (acc == ref_partial(row, k));                                  // I2
    assert (hi_rem == ref_hi(row, k) && lo_rem == ref_lo(row, k));       // I3
    assert (acc + lo_rem <= ref_y(row) && ref_y(row) <= acc + hi_rem);   // I4
    if (k == P) assert (hi_rem == 0 && lo_rem == 0);                      // I6
end
always @(*) if (f_valid && done && mode == MODE_RELU)
    assert (y_out == (ref_y(row) > 0 ? ref_y(row) : 0));                  // D1
always @(posedge clk) if (f_valid && $past(f_valid) && $past(busy) && busy && k == $past(k) + 1) begin
    assert (acc + hi_rem <= $past(acc + hi_rem));                         // I5
    assert (acc + lo_rem >= $past(acc + lo_rem));
end
always @(*) if (f_valid && tbl_ready) begin                               // I1
    assert (mask_sum(f_mask) <= T[popcount(f_mask)]);
    assert (mask_sum(f_mask) >= T[N] - T[N - popcount(f_mask)]);
end
```

### 7.3 Proof plan and instance size

- **Small instance:** `N=4`, `XB=3`, `P=4` (int4 weights), `R=3` rows, `MB=3`. Run the 2C, SF and
  SM modes as separate tasks.
- **Width:** the values satisfy $|A_k + \mathrm{Hi}_k|, |A_k + \mathrm{Lo}_k| \le 2^{P-1} S_{\max}$
  with $S_{\max} = N(2^{XB} - 1)$. So `AW >= P + clog2(S_max)`, plus bias headroom: 9 bits for the
  small instance and 22 for `P=8`, `N=64`, `XB=8`. Use 12 bits in the harness and let I7 check the
  margin.
- **Solver load:** the free data is about 64 bits (12 bits of activations, 48 of weights, 4 of
  mask), which is small for yices or boolector.
- **Bounded model check:** depth at least `P*(w_p+1)+4` from reset for D1–D4.
- **Unbounded proof:** k-induction with I1–I8 asserted; depth 2–4 should suffice, because I2 and
  I3 make the datapath state a function of `(row, k)`.
- **Parameter sweep:** repeat with `N=8`, `P=8` as a stretch goal. Theorems 1 and 2 are
  parameter-generic, but the bit-blasted proof is not.

---

## 8. Numerical verification (`model/theory_check.py`)

Run `python3 model/theory_check.py`; it takes about 37 s with numpy only. The script exits with
status 0 iff every check passes. Use `--quick` for a run of about 11 s. Last full run: **ALL CHECKS
PASSED**.

| check | instances | result |
|---|---|---|
| Theorem 1: every code word × every $k$ × 40 vectors (ties, zeros, edge cases) | 2C P4N3 819,200 and P3N4 655,360; SF 819,200 and 655,360; SM 819,200; SD P3N3 3,149,280 and P4N2 1,312,200; NAF P4N3 819,200 and P5N2 245,760 | 0 violations |
| Theorem 2(a): max and min over each (prefix, popcount) class | 2C, SD | equal to the bound in every class |
| Theorem 2(b): restricted-statistics bound | SF, SM | exact in every class; global bound strictly loose on 23–31 % of sides |
| Theorem 2(c): max over all permutations of $x$ | SF, SM | equal to the global bound in every class |
| Theorem 2(d): NAF | | sound; strictly loose in 29–47 % of cases; counterexample (30 vs 21) reproduced |
| Proposition 4.1: tangent inequalities, both branches, every $(m,n)$ | 917,648 | all hold |
| Proposition 4.2: envelope sound, exact on $M$, gap ≤ bound | 600 random $x$ | worst gap/bound 0.68 |
| Proposition 4.3: rounding, $p = 0..3$ | | excess 0.98, 0.49, 0.24, 0.12 ≤ 1, 1/2, 1/4, 1/8 |
| Propositions 4.4 and 4.5 | random | hold |
| Corollaries 2.1/2.2, Proposition 2.4, Proposition 5.1 (ReLU and random monotone staircases), all 8 schemes × 5 codes, exhaustive rows | 18,466,304 outputs | bit-exact, nested, exact at $k = P$ |
| Proposition 2.5: 2C and SF B1 boxes | exhaustive int3/int4 | identical |
| Propositions 5.2/5.3: argmax with elimination, ties forced | 3,000 small instances plus 4,000 layer instances | always `== np.argmax` |
| Theorem 3: $K \le \bar{K}$ for ReLU and 4-bit requant, on every row of every scenario | about 172k row checks | holds |
| Corollary 6.1: tail-sum identity | Monte-Carlo and layers | exact |
| Proposition 6.2 closed form, $\lambda$ = 0.5, 2, 8, 32 | Monte-Carlo, 200k | 7.930/7.931, 7.569/7.570, 6.922/6.919, 6.089/6.076 (within 2.5 σ) |
| Proposition 6.2 constant $C$ | | 1.33274 numeric vs 1.33275 = γ/ln 2 + ½ |
| Proposition 6.3 closed form, $\Delta/\omega$ = 0.5, 4, 32 | Monte-Carlo, 200k | 8.000/8.000, 7.497/7.500, 5.624/5.625, all below $k_\Delta + 2$ |
| Proposition 6.4 break-even | 16 configurations × 2 metadata schemes, $N$ = 64 and 256 | an exact accounting identity; the sign of (saving − $\mu/N$) matches the winner in all 32 comparisons |

---

## 9. Verdict, limitations, next steps

**What is solid:**

- The interval certificate is sound for every code, tight where Theorem 2 says, and
  information-optimal for 2C (Corollary 3.1).
- The decision layer generalizes MSB-first "early negative detection" in three ways: (i) any
  monotone integer requantization, including the zero bucket, saturation and interior cells;
  (ii) argmax with elimination under any schedule; (iii) stickiness, which tolerates pipelining.
  All are bit-exact and backed by the invariant list of section 7.
- The cost law (Theorem 3, Corollary 6.1, Proposition 6.2) predicts the measured plane counts
  within 1 to 2 %.

**What does not hold up:**

- The claim that popcount metadata beats the metadata-free box baseline in general fails. On
  dense int8 grids it is worse, because of the metadata words and because of P3. The B1 box itself
  is the main source of savings: 17 % of words at 48 % negative rows, 28 % at 70 %, and 46 % for
  argmax.
- The prior work cited in the brief (e.g. SnaPEA ISCA'18 exact mode, ComPEND) is described there as
  close to B1. B1 is therefore not a novelty claim. This note does not include a literature search.
  Statements such as "the popcount/order-statistic interval and its tightness theorem", "the
  argmax certificate with elimination for bit-serial engines" or "the requant-cell stop" must be
  checked against the literature before they are called new.

**Recommendations for the design:**

1. Build the certificate datapath generically: the interval registers, a comparator against a
   monotone $f$, and the argmax relation $\rhd$. It is cheap and provably exact.
2. Make metadata optional per layer. The offline compiler should enable it only when Proposition
   6.4 holds on calibration data.
3. Choose SM over SF/2C for power-of-two-like quantizers.
4. Prefer large-$N$ tiles, which amortize the metadata.

**Open questions that could change the picture:**

- Real trained layers have stable per-channel activation statistics. That could make Proposition
  4.5 (calibration-ordered chunks with their own tables) much tighter than on i.i.d. synthetic $x$.
- Low-bit-density weight codes beyond powers of two.
- Real negative fractions after BatchNorm, which are often above 50 %; B1 savings scale with $q$
  (P2).

All three must be measured on real layers before any SOTA claim. The synthetic figures above are
illustrations, not results.
