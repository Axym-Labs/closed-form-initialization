# Residual BT-Gradient Trust-Region Check

## Mechanism Tested

The non-ad-hoc correction was to keep the residual BT-gradient moment solve,
but put nondestructiveness inside the normal equations instead of applying it
as a post-hoc cap.

For a LayerNorm residual layer

\[
H^+ = \operatorname{LN}(H + \Phi B),
\]

the added term penalizes the first-order LayerNorm-manifold displacement:

\[
\min_B
\|\mathcal A(B) + \eta \nabla_C L_{\rm BT}\|_F^2
+ \mu
\frac{1}{|\mathcal S|}
\sum_{s\in\mathcal S}
\|J_{\operatorname{LN},H_s}\Phi_s B\|_F^2
+ \rho\|B\|_F^2.
\]

Here \(\mathcal A(B)\) is the existing linearized BT-correlation operator and
\(\mathcal S\) was the two augmented views plus the clean/base stream. The
kinetic term is operator-normalized against the BT moment operator, so
\(\mu=1\) means equal random-probe operator energy rather than an arbitrary
sample-space unit.

## Result

The term was implemented as a new `layernorm_sample` operator and passed an
adjoint identity check on CUDA (`abs_err = 8.3e-7`).

On full CIFAR100/SimCLR, width 512, depth 12, K=4 moment batches, LayerNorm
residual, random nonlinear branch, post-activation paired CF transform, and
exact BT-quadratic scaling:

| Variant | Train BT | Test BT | Improve frac | Rank | Agreement-rank | Update | Last acc | All-PCA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no kinetic | 0.3846 | 0.4002 | 1.0 | 15.64 | 64.38 | 0.262 | 0.1662 | 0.1814 |
| kinetic \(\mu=1\) | 0.3821 | 0.3978 | 1.0 | 15.50 | 60.74 | 0.236 | 0.1650 | 0.1830 |

The kinetic term changed the solve in the intended direction: actual
post-LN update decreased and actual-vs-linearized correlation motion improved
(`0.864 -> 0.882`). But it did not repair the spectral narrowing or last-layer
representation quality. Therefore the failure is not simply excessive
LayerNorm-manifold step size.

## Branch-Metric Boundary

I then tested whether the paired CF post-transform itself was the spectral
narrowing source by replacing it with post-activation whitening only. This
kept more breadth (`rank 45.83`, agreement-rank around `213-221`) but the
full-train quadratic BT check vetoed every layer: the realized first-order BT
term was positive, about `+0.087` to `+0.089` per layer, so the scale was zero.

A weaker paired-shrink transform (`branch_post_invariance = 0.25`) was also
fully vetoed. Thus the current branch metric has a hard feasibility tradeoff:

- broad/no-shrink nonlinear features are not globally BT-descending under the
  realized residual LayerNorm map;
- sufficiently strong paired CF shrink gives monotone BT descent;
- that same shrink concentrates the paired-difference spectrum and yields
  low-rank representations.

Compared with greedy residual BP-BT, this is the crucial mismatch. Greedy
BP-BT keeps the agreement-difference spectrum broad while improving BT:
agreement effective rank stays around `360-475` over depths 12/24, whereas
the current CF residual flow falls to roughly `60-70`.

## Interpretation

The exact failure is now more specific than the previous scans:

\[
\text{BT descent} \cap \text{broad nonlinear branch cone}
\]

is not being found by the current branch metric. The post-activation paired
CF transform supplies descent by preselecting agreement directions, but it
preselects too narrowly. Whitening supplies breadth, but the induced residual
BT-gradient projection points uphill on the full train distribution.

The next viable formulation should choose the branch metric itself by a
descent-constrained breadth principle, not add another downstream filter. A
natural target is:

\[
\max_P \operatorname{breadth}(P)
\quad\text{subject to}\quad
\langle \nabla_C L_{\rm BT}, \Delta C(B^\star(P))\rangle < 0
\]

with \(B^\star(P)\) still given by the residual moment-OLS solve. This would
turn the paired-shrink strength into a feasibility variable: use the least
destructive branch metric that produces a globally descending residual
BT-gradient step.
