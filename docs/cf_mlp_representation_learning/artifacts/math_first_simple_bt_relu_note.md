# Math-First CF-BT Mechanism Note

## Simplified Object

Consider one scalar preactivation coordinate for two positive views:

\[
u = s + d,\qquad v = s - d,
\]

where \(s\) is the shared component and \(d\) is the view-difference component.
BT wants the post-layer coordinates to have high diagonal correlation and low
off-diagonal correlation. For one coordinate, the diagonal part is controlled
by

\[
\operatorname{corr}(\phi(u), \phi(v)),
\]

after centering and scaling. The off-diagonal part is a separate covariance
decorrelation problem.

## Basic Lemma

If a coordinate has low positive-pair correlation before the nonlinearity, a
per-view ReLU threshold cannot make it a high-correlation BT diagonal
coordinate.

For zero-mean jointly Gaussian \(u,v\) with correlation \(\rho\),

\[
\mathbb E[\operatorname{ReLU}(u)\operatorname{ReLU}(v)]
=
\frac{\sqrt{1-\rho^2}+(\pi-\arccos\rho)\rho}{2\pi}.
\]

After subtracting the ReLU mean and dividing by the ReLU variance, this gives
a monotone function of \(\rho\). In particular, if \(\rho=0\), then
\(\operatorname{ReLU}(u)\) and \(\operatorname{ReLU}(v)\) are independent and
their centered correlation is still \(0\). Thresholding harder can remove
variance, but it does not create shared signal.

So the simple failure mode is:

\[
\text{clip low-agreement modes}
\quad\Rightarrow\quad
\text{little remaining paired covariance}
\quad\Rightarrow\quad
\text{CCA can decorrelate axes but cannot recover diagonal alignment}.
\]

## Consequence

The intended operation should not be "send low-agreement directions negative."
That kills or sparsifies the whole scalar coordinate, including whatever
shared component \(s\) it contains. The mathematically cleaner target is:

\[
\text{preserve } \operatorname{Var}(s)
\quad\text{while reducing}\quad
\operatorname{Var}(d),
\]

then place the resulting preactivation in a nonlinear region where the two
views cross similar gates. In other words, the nonlinearity should act on an
already-aligned common signal, not be asked to manufacture alignment from a
low-correlation coordinate.

A simple next parametrization target is therefore a shared-signal-metric
shrinkage:

\[
\Sigma_s=\operatorname{Cov}\left(\frac{x_1+x_2}{2}\right),\qquad
\Sigma_d=\operatorname{Cov}\left(\frac{x_1-x_2}{2}\right),
\]

with directions ranked by the generalized ratio

\[
\Sigma_d v = \mu(\Sigma_s+\epsilon I)v.
\]

Shrink large-\(\mu\) directions while preserving the \(\Sigma_s\)-metric. This
is simpler than clip-then-CCA: it directly states what should survive and what
should be suppressed before the activation.

## Validation From Existing Runs

Depth-24 full-data stage diagnostics already show the predicted split.

| Variant | Final BT/dim | On diag/dim | Weighted off/dim | Corr diag | Pre->act novelty |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain ReLU CF | 0.5893 | 0.5272 | 0.0621 | 0.285 | 0.0083 |
| agreement bias | 0.7463 | 0.7317 | 0.0146 | 0.187 | 0.1292 |
| active-rank clipping | 0.8029 | 0.7892 | 0.0136 | 0.123 | 0.1822 |
| corr-bias clipping | 0.8117 | 0.8006 | 0.0111 | 0.123 | 0.1892 |
| shared-CCA only, no TF32 | 0.1980 | 0.1464 | 0.0516 | 0.639 | 0.1259 |
| active-rank + CCA, no TF32 | 0.9095 | 0.9092 | 0.0003 | 0.047 | 0.1858 |
| corr-bias + CCA, no TF32 | 0.7741 | 0.7711 | 0.0030 | 0.151 | 0.0960 |

The clipping variants do create nonlinear novelty and reduce covariance
error, but the BT loss becomes almost purely diagonal-alignment failure.
Adding CCA after clipping makes the off-diagonal term tiny, but diagonal
alignment collapses further. This is exactly what the scalar argument predicts:
the nonlinear clipping step removed paired covariance that a later linear
decorrelator cannot recreate.

## Simplicity-Biased Decision

Do not add more active-rate or corr-bias schedules right now. They are variants
of the wrong scalar operation. The next useful validation is a direct
\((s,d)\)-decomposition diagnostic: measure whether a proposed layer preserves
\(\operatorname{tr}\Sigma_s\) while reducing \(\operatorname{tr}\Sigma_d\)
before and after the activation. Only if that diagnostic improves should a new
parametrization be tested downstream.

## Shared/Difference Diagnostic Result

I added `cf_mlp_shared_difference_diagnostic.py` and ran it on the depth-24
full-data setting. The diagnostic confirmed the criterion:

| Variant | Final BT/dim | Corr diag | Final shared/diff | Best shared/diff | Best layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain ReLU CF | 0.5468 | 0.3131 | 1.916 | 2.795 | 1 |
| agreement bias | 0.7474 | 0.1864 | 1.483 | 1.483 | 24 |
| active-rank clipping | 0.8041 | 0.1217 | 1.354 | 1.354 | 24 |
| corr-bias clipping | 0.8071 | 0.1269 | 1.434 | 1.434 | 24 |
| shared-CCA only, no TF32 | 0.1980 | 0.6392 | 4.773 | 9.265 | 12 |
| active-rank + CCA, no TF32 | 0.9095 | 0.0471 | 1.102 | 1.102 | 24 |
| corr-bias + CCA, no TF32 | 0.7741 | 0.1512 | 1.632 | 1.665 | 23 |

This says the same thing in the simpler variables: the rejected clipping
variants never build a large shared/difference ratio; they mostly lower both
shared and difference traces. The only current CF path that builds a large
ratio is shared-CCA, and its ratio peak coincides with the good BT region.

I also tested the simplest shared-metric shrinkage,
`plain_cf_sharedmetric_relu`, where the CF shrinkage is formed in the
\(\Sigma_s\) metric instead of the total covariance metric. It is rejected as
too naive: at depth 24 it ends worse than plain ReLU on the intended
diagnostic and on BT (`shared/diff = 1.774` vs `1.916`, BT/dim `0.5755` vs
`0.5468`). So merely changing the shrinkage metric is not enough; the next
candidate must preserve shared coordinates through activation and
normalization, not just alter the linear shrinkage basis.

## Constant Invariance Check

The default nonresidual ReLU CF schedule was also tested as a possible
confounder. In code, `relax4x` means invariance strength decays by 4x per
layer. Removing it with `plain_cf_relu_constinv1.0` made the layerwise BT
trajectory worse:

| Setup | Depth | Final BT/dim | Best BT/dim | Best layer | Final corr diag |
| --- | ---: | ---: | ---: | ---: | ---: |
| default relax4x | 6 | 0.5344 | 0.5291 | 4 | 0.355 |
| default relax4x | 12 | 0.5581 | 0.5291 | 4 | 0.320 |
| default relax4x | 24 | 0.5893 | 0.5291 | 4 | 0.285 |
| constant invariance | 6 | 0.6176 | 0.5483 | 1 | 0.355 |
| constant invariance | 12 | 0.7033 | 0.5483 | 1 | 0.246 |
| constant invariance | 24 | 0.7301 | 0.5483 | 1 | 0.208 |

The shared/difference diagnostic agrees. At depth 24, constant invariance
ends with shared/diff `2.339` and peaks at only `2.868` on layer 2. Its
preactivation-to-postactivation ratio gain is `0.926`, so the activation
stage is not preserving the desired shared-over-difference geometry.

This rejects the "relax4x caused the plateau" explanation. The schedule is
acting more like damage control: it weakens later local updates after the
early ReLU CF step has already reached its useful range. With constant
invariance, the same non-compositional update is applied strongly at every
layer and diagonal view alignment degrades faster.

## Agreement-Subspace Expansion

The next failure in the "send bad modes negative" idea is that BT evaluates
every output coordinate after per-coordinate standardization. A coordinate that
is merely shrunk still reappears in the BT objective unless it is almost
constant, and a constant coordinate contributes on-diagonal error. So the
output coordinate must be replaced by a useful high-agreement feature, not just
scaled down.

The tested replacement is:

\[
M = \Sigma^{-1/2}\Delta\Sigma^{-1/2},\qquad
M q_i = \mu_i q_i,\qquad \mu_1 \leq \mu_2 \leq \cdots,
\]

where \(\Delta=\operatorname{Cov}(x_1-x_2)\) and \(\Sigma\) is the average
view covariance. Keep the first \(k\) high-agreement modes and expand them
back to width \(d=512\):

\[
T_k = \Sigma^{-1/2} Q_{1:k} R_{k\times d},
\]

with fixed deterministic mixed columns \(R\). This explicitly replaces
low-agreement output coordinates with nonlinear features of the
high-agreement subspace.

Expansion alone solves diagonal alignment but creates redundant coordinates:

| Variant | Depth 6 | Depth 12 | Depth 24 | Depth-24 corr diag | Depth-24 weighted off/dim |
| --- | ---: | ---: | ---: | ---: | ---: |
| expand k128 | 0.2084 | 0.2693 | 0.2651 | 1.000 | 0.2651 |
| expand k192 | 0.3554 | 0.2405 | 0.2639 | 1.000 | 0.2639 |
| expand k224 | 0.4629 | 0.2076 | 0.2612 | 1.000 | 0.2612 |
| expand k256 | 0.5612 | 0.2370 | 0.2533 | 0.992 | 0.2533 |

Adding full whitening after the expansion removes this redundancy once the
positive-pair alignment has been formed. The strongest tested fixed point is
`plain_cf_agreement_expand_fullwhiten_relu_k192`:

| Depth | Final BT/dim | Best layer | Final corr diag | On/dim | Weighted off/dim |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.5384 | 6 | 0.267 | 0.5378 | 0.0006 |
| 12 | 0.2456 | 12 | 0.509 | 0.2429 | 0.0027 |
| 24 | 0.0083 | 24 | 0.972 | 0.0009 | 0.0074 |

The depth-24 shared/difference diagnostic confirms the mechanism:
BT/dim `0.008264`, corr-diag `0.9718`, shared/diff `71.07`, diff fraction
`0.01388`, best layer `24`.

This is the first tested nonresidual CF-BT mechanism that genuinely makes
depth work for the training-pair BT objective. It does not solve representation learning:
the same k192 fullwhiten representation has last-layer CIFAR100 probe accuracy
`0.0520/0.0134/0.0186` for depths `6/12/24`, all-layer PCA512
`0.1028/0.0932/0.0886`, and best individual layer remains layer 1 at `0.1384`.
So the plateau cause and the semantic failure are now separated: the old CF
depth path failed to replace low-agreement output coordinates; the repaired BT
path can become too invariant/label-poor.

## Held-Out Pair and Classification Guardrails

A later train/test paired-view diagnostic makes the limitation sharper. Plain
ReLU has nearly matching train/test BT, so its flat trajectory is a real
mechanistic plateau rather than train-pair overfitting:

| Variant | Depth | Train BT/dim | Test BT/dim | Test corr diag |
| --- | ---: | ---: | ---: | ---: |
| plain ReLU | 24 | 0.5467 | 0.5447 | 0.316 |
| expand fullwhiten k192 | 24 | 0.0083 | 0.9901 | 0.006 |
| expand-only k128 | 24 | 0.2651 | 0.9945 | 0.003 |
| expand-only k192 | 24 | 0.2639 | 0.9267 | 0.041 |
| agreement+shared-CCA | 24 | 0.5368 | 1.0240 | -0.003 |

So agreement-subspace expansion and the older CCA repair fit the finite
training augmented pairs; they do not learn a held-out positive-pair
invariance.

The downstream CIFAR100 readout sweep across cutting strength confirms that
the cut/replace mechanism is too aggressive for classification:

| Variant family | Depth 24 final-layer acc | Depth 24 all-layer PCA512 | Best layer |
| --- | ---: | ---: | ---: |
| plain ReLU | 0.1094 | 0.1508 | 0.1704 |
| expand-only best over k | 0.0174 | 0.1204 | 0.1384 |
| expand fullwhiten best over k | 0.0186 | 0.1008 | 0.1434 |
| expand fullwhiten k192 | 0.0186 | 0.0886 | 0.1384 |

This means the cutting mechanism should be treated as a useful negative
control. It demonstrates that replacing low-agreement dimensions can optimize
the training BT geometry, but it also overfits positive-pair covariance and
removes class-relevant information. Future variants need a generalizing
nonlinearity-aware objective or regularizer, not stronger agreement-basis
cutting.
