# LayerNorm Residual Velocity And Branch Dictionary Diagnostic

## Motivation

The previous branch-filter scan was the wrong level of intervention. The
residual BP-BT control uses

\[
H_{l+1}=\operatorname{LayerNorm}\left(H_l+\phi(H_lW_l)\right),
\]

whereas the residual moment-OLS CF solver had been applying a feature-wise
dataset normalization after the residual addition. The mathematically natural
repair is to put the closed-form residual velocity on the same row-wise
LayerNorm manifold as BP-BT.

For one sample, with

\[
y=\frac{x-\mu(x)}{\sigma(x)},
\]

the LayerNorm tangent for a perturbation \(u\) is

\[
T_xu=
\frac{u-\bar u\mathbf 1-y\,\langle y,u-\bar u\mathbf 1\rangle_d}{\sigma(x)}.
\]

The corrected residual moment operator for branch features
\(\Phi_i=\phi(H_iA)\) is therefore

\[
\mathcal A_{\mathrm{LN}}(B)
=
\frac1n
\left[
S_1(T_{H_1}\Phi_1B)^\top Z_2
+Z_1^\top S_2(T_{H_2}\Phi_2B)
\right],
\]

where \(S_i\) is the existing batch-standardization tangent used by the BT
correlation objective. The closed-form step still solves

\[
B^\star=\arg\min_B
\|\mathcal A_{\mathrm{LN}}(B)+\eta\nabla_C L_{\mathrm{BT}}\|_F^2
+\rho\|B\|_F^2.
\]

The trust region was also moved to the post-retraction motion,

\[
\frac{\|\operatorname{LayerNorm}(H+\Phi B)-H\|_{\mathrm{rms}}}
{\|H\|_{\mathrm{rms}}},
\]

instead of the raw branch norm. The first tested cap, `0.75`, was anchored to
the greedy residual BP-BT post-normalization update scale.

## Results

Full CIFAR100/SimCLR, width 512, branch width 512, moment batch 1024,
`ols_ridge=1e-5`, `cg_iters=120`, seed 7:

| Setup | Depth | Train BT | Test BT | Improve frac | Rank | Target cos | Actual-pred cos | Post-LN update | Update/input cos | Last acc | All-PCA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| feature CF old-span adaptive | 12 | 0.3418 | 0.3603 | 1.00 | 23.7 | 0.186 | 0.765 | n/a | n/a | 0.1720 | 0.1934 |
| LayerNorm + CF-shrink branch | 12 | 0.6655 | 0.7061 | 0.17 | 96.5 | 0.388 | 0.341 | 0.750 | -0.371 | 0.1742 | 0.1902 |
| LayerNorm + random branch | 12 | 0.9032 | 0.9361 | 0.00 | 196.7 | 0.639 | 0.234 | 0.750 | -0.374 | 0.1066 | 0.1598 |
| LayerNorm + random branch + post-activation CF transform | 12 | 0.5910 | 0.6301 | 0.75 | 40.9 | 0.272 | 0.225 | 0.750 | -0.370 | 0.1804 | 0.1902 |
| LayerNorm + random branch + grad-reach transform, cap 0.25 | 12 | 0.6784 | 0.7039 | 0.00 | 159.9 | 0.802 | 0.428 | 0.250 | -0.124 | 0.1660 | 0.1710 |
| LayerNorm + random branch + post-activation CF, K=4 + BT-quadratic scale | 12 | 0.3846 | 0.4002 | 1.00 | 15.6 | 0.101 | 0.864 | 0.262 | -0.129 | 0.1662 | 0.1814 |
| LayerNorm + random branch + post-activation CF, K=4 + BT-quadratic scale | 24 | 0.3707 | 0.3845 | 1.00 | 14.2 | 0.083 | 0.971 | 0.169 | n/a | 0.1632 | 0.1824 |

Depth scaling for LayerNorm + CF-shrink branch:

| Depth | Train BT | Test BT | Improve frac | Rank | Last acc | All-PCA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.6221 | 0.6427 | 0.17 | 78.0 | 0.1828 | 0.1888 |
| 12 | 0.6655 | 0.7061 | 0.17 | 96.5 | 0.1742 | 0.1902 |
| 24 | 0.4848 | 0.5106 | 0.71 | 46.3 | 0.1742 | 0.1922 |

## Interpretation

LayerNorm fixes the wrong part by itself. It gives a BP-like post-normalization
update signature: large residual motion, negative update/input row cosine, and
much higher covariance rank. But it does not produce the desired BP-like BT
trajectory. At d6/d12, most layers worsen BT; at d24 the trajectory is more
often improving but remains far weaker than the old feature-normalized
moment-OLS baseline.

The branch dictionary diagnostic is more specific:

- `random_orth` has much higher linearized target reach (`target cos` about
  `0.64`) but destroys invariance and semantics.
- `cf_shrink` is paired/stable but too weak a velocity basis.
- applying a closed-form paired transform after the random nonlinear lift
  partially repairs random features (`test BT 0.9361 -> 0.6301`) and recovers
  the readout to the LayerNorm CF-shrink level, but still does not beat the
  old feature-normalized CF baseline.

So the exact current failure is not simply normalization mismatch, rank
collapse, or raw step size. It is the branch dictionary problem:

\[
\text{we need nonlinear directions that are simultaneously BT-gradient
reachable and paired-view invariant.}
\]

Random features satisfy reach but not invariance. CF-shrink features satisfy
some invariance but not reach. Post-activation CF is the right structural
direction, because it places the activation between two linear maps, but the
simple version is still not enough.

## Next Mechanism

The next candidate should not be another filter. It should choose the branch
dictionary by a joint criterion before solving \(B\):

\[
A^\star \quad\text{or branch transform }P^\star
\quad\text{should maximize paired stability and projected BT-gradient reach.}
\]

A natural closed-form objective is to construct nonlinear random/lifted
features \(\Phi\), then solve for a post-activation transform \(P\) whose
columns score highly under both:

1. low paired-view difference in \(\Phi P\);
2. high contribution to the adjoint BT-gradient energy
   \(\|\mathcal A_{\mathrm{LN}}^\ast(G)\|^2\).

This would directly combine the two statistics separated by the diagnostic,
instead of selecting subspaces after the fact or adding an old-span penalty.

## Grad-Reach Falsification

The first direct version of that idea was also tested. It forms

\[
R=\mathcal A_{\mathrm{LN}}^\ast(G)\mathcal A_{\mathrm{LN}}^\ast(G)^\top
\]

in branch-feature space and chooses post-activation branch coordinates by the
generalized ratio

\[
\max_p
\frac{p^\top \Sigma_\Phi^{-1/2}R\Sigma_\Phi^{-1/2}p}
{p^\top(\Sigma_\Phi^{-1/2}\Delta_\Phi\Sigma_\Phi^{-1/2}
+\lambda I)p}.
\]

This is the obvious "high BT-gradient reach per paired-difference" rule. It
failed. With the BP-scale post-LN cap `0.75`, the depth-1 step maximized
linearized target reach (`target cos 0.726`) but the realized movement was
almost unrelated to the predicted one (`actual-pred cos 0.077`) and worsened
BT. Reducing the post-LN cap to `0.25` made the local step coherent
(`actual-pred cos 0.462`, train BT improved `0.5793 -> 0.5696`), but the full
d12 trajectory still moved BT the wrong way at every layer
(`0.6784/0.7039`, improve fraction `0.00`).

So "reach divided by invariance cost" is not sufficient. It selects directions
that have high infinitesimal BT-gradient leverage but whose repeated finite
LayerNorm-retracted use destroys the paired representation. The missing term
is a curvature/compositionality object: a branch direction must not only have
positive first-order reach, it must remain stable under the nonlinear
retraction and under repeated recomputation across layers.

## Curvature-Scaled Post-Activation CF

The failed grad-reach run revealed a more precise issue. On the fit batch, the
first-order BT term was negative, but on the full training distribution the
same finite candidate often had a positive first-order BT change. I added an
analytic scale using the exact quadratic form of BT in correlation space.
For the realized finite correlation displacement \(D\),

\[
L(C+\alpha D)-L(C)
=
\alpha\langle G,D\rangle+\alpha^2Q(D),
\]

so

\[
\alpha^\star=
\operatorname{clip}_{[0,1]}
\left(-\frac{\langle G,D\rangle}{2Q(D)}\right),
\]

with \(\alpha^\star=0\) when the realized full-train first-order term is not
descending.

This correctly vetoed the naive `grad_reach` branch: its full-train
first-order term was positive on every d12 layer, so the analytic scale chose
zero. Increasing the moment equation to K=4 independent batches did not fix
that. The high-reach dictionary is therefore globally wrong, not merely
over-stepped.

The post-activation CF branch behaved differently. With one 1024 batch, the
quadratic guard also vetoed it. With K=4 independent moment batches, the full
train realized first-order term became consistently negative and the
quadratic scale stayed at 1. This gives the first coherent LayerNorm residual
CF-BT trajectory:

- d12: train/test BT `0.3846/0.4002`, improvement fraction `1.0`,
  actual-predicted cosine `0.864`.
- d24: train/test BT `0.3707/0.3845`, improvement fraction `1.0`,
  actual-predicted cosine `0.971`.

This is a real mechanism-level improvement over the earlier LayerNorm
attempts: residual LayerNorm, nonlinear lift, post-activation paired CF
transform, multi-batch moment solve, and exact BT-curvature scaling produce a
monotone and composing BT trajectory.

But it still does not solve the representation problem. The trajectory
narrows with depth: d24 rank is only `14.2`, all-PCA is `0.1824`, and the
last-layer readout is `0.1632`, below the old feature-normalized adaptive
old-span baseline (`d24 test BT 0.3334`, rank `22.2`, all-PCA `0.1952`).

The new failure is now cleaner:

\[
\text{we can get a correct LayerNorm residual BT flow, but its stable
post-activation CF branch still allocates too few useful modes.}
\]

The next mechanism should keep this curvature-scaled residual-flow structure
and modify only the branch allocation so that monotone BT does not imply rank
collapse. The likely object is not raw reach; it is a multi-batch,
full-train-descending branch basis with an explicit mode allocation floor or
diversity constraint inside the post-activation paired transform.
