# BP Moment Velocity And Balanced-Identity CF

## Question

The previous branch/gain scans were not changing the core mechanism enough. The
proper route-3 question is:

\[
\Delta C_l^{\mathrm{BP}}
= C_{l+1}^{\mathrm{BP}}-C_l^{\mathrm{BP}}
\]

for residual BP-BT layers. Which simple moment-space law does this velocity
follow?

## BP Moment Velocity Diagnostic

For each saved residual BP-BT layer, I measured the realized cross-correlation
velocity and compared it to three candidate laws:

\[
-\nabla_C L_{\mathrm{BT}},\qquad I-C,\qquad \operatorname{polar}(C)-C.
\]

The result is decisive: residual BP-BT is not well described as an Euclidean
BT-gradient step in correlation coordinates. It is much closer to an identity
or polar correction of the whole cross-correlation operator.

| Model | Depth | Final BT/dim | Step improve frac | Mean cos \(\Delta C,-\nabla L\) | Mean cos \(\Delta C,I-C\) | Mean cos \(\Delta C,\operatorname{polar}(C)-C\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e residual BP-BT | 6 | 0.1482 | 0.833 | 0.037 | 0.663 | 0.640 |
| e2e residual BP-BT | 12 | 0.0795 | 0.833 | 0.033 | 0.654 | 0.634 |
| e2e residual BP-BT | 24 | 0.1198 | 0.917 | 0.035 | 0.638 | 0.630 |
| greedy residual BP-BT | 6 | 0.1857 | 1.000 | 0.034 | 0.687 | 0.673 |
| greedy residual BP-BT | 12 | 0.0695 | 1.000 | 0.034 | 0.651 | 0.643 |
| greedy residual BP-BT | 24 | 0.0297 | 1.000 | 0.031 | 0.623 | 0.618 |

This also strengthens the earlier cross-layer-credit interpretation. Greedy
residual BP-BT, with no end-to-end credit assignment, has the same moment-law
signature as e2e residual BP-BT.

Artifact:

- `docs/cf_mlp_representation_learning/artifacts_bp_moment_velocity_seed7/`

## Derived CF Update

The direct \(I-C\) target failed under the current moment-OLS layer. The reason
was not that \(I-C\) is non-descending; algebraically, at the input

\[
\langle \nabla_C L_{\mathrm{BT}}, I-C\rangle/d < 0.
\]

The failure was a fitting-geometry problem. In Frobenius norm, the off-diagonal
block has \(d(d-1)\) entries and dominates the \(d\)-entry diagonal block. The
OLS solve matched off-diagonal identity motion while moving the diagonal in the
wrong direction. On a smoke run, raw `identity` with the current branch included
had target cosine about `0.94`, but the achieved diagonal delta was negative
and BT worsened.

The derived target is therefore a block-balanced identity correction:

\[
T_{\mathrm{bal}}(C)_{ij} =
\begin{cases}
\sqrt{d-1}(1-C_{ii}) & i=j,\\
-C_{ij} & i\ne j.
\end{cases}
\]

This is not a branch filter. It changes the core moment target so the diagonal
and off-diagonal parts of the correlation objective have comparable aggregate
weight. The residual branch dictionary is also changed in the natural residual
way:

\[
\Phi(H) = [H,\ \phi(HA)].
\]

This includes the linear Sylvester-like residual tangent \(HB_0\) while keeping
the nonlinear branch \(\phi(HA)B_1\) in the layer.

## CIFAR100/SimCLR Validation

All runs use width 512, current-plus-random nonlinear branch, LayerNorm
residual update, K=4 moment batches of size 1024, ridge `1e-5`, LayerNorm
kinetic weight `1`, exact BT quadratic scaling, and post-LayerNorm trust cap
`0.25`.

| Depth | Train BT | Test BT | Improve frac | Corr diag train/test | Rank | Last acc | All-PCA acc | Best layer |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 6 | 0.5435 | 0.5768 | 1.000 | 0.276/0.253 | 45.8 | 0.1704 | 0.1776 | 0.1704 @ 6 |
| 12 | 0.5037 | 0.5505 | 1.000 | 0.298/0.266 | 51.7 | 0.1840 | 0.1888 | 0.1840 @ 12 |
| 24 | 0.4527 | 0.4934 | 1.000 | 0.338/0.308 | 42.7 | 0.1818 | 0.1946 | 0.1818 @ 22 |

Layerwise verification found zero train-BT non-improving steps and zero
test-BT increase steps for all three depths.

Representation content also moves in the right direction with depth. For d24,
layer 1 to layer 24 changes:

- class linear accuracy: `0.1546 -> 0.1818`
- CKA to labels: `0.047 -> 0.059`
- view retrieval top1: `0.089 -> 0.123`
- raw reconstruction R2: `0.901 -> 0.864`
- CKA to raw input: `0.469 -> 0.345`

Artifacts:

- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_balanced_identity_current_cap025_k4_d12_b1024_ridge1e5/`
- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_balanced_identity_current_cap025_k4_d6d24_b1024_ridge1e5/`

## Interpretation

This is not yet competitive with BP-BT. BP-BT reaches much lower BT objective
values and better all-layer representations. It is also not strictly better
than the previous progress-fraction gain-floor CF in readout.

But it is a cleaner mechanism:

1. It is derived from BP's measured layerwise moment velocity.
2. It changes the core target and residual dictionary, not a post-hoc selector.
3. It produces a BP-like monotone trajectory across depth.
4. Last-layer representation quality improves into later layers, especially
   from d6 to d12/d24.

The current exact gap is magnitude and generalization of the moment flow. The
balanced target gives the correct trajectory shape, but the realized flow is
much weaker than BP. The next mathematical object should be a better
normalization-aware residual operator for realizing balanced \(I-C\) velocity,
not more branch filters.
