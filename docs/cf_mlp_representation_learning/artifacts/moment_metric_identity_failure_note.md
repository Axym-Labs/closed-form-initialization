# Moment-Metric Identity Diagnostic

## Question

The BP-BT velocity diagnostic showed that residual BP-BT layer updates are
better described by broad identity/polar correction,

\[
\Delta C_l \approx I - C_l \quad \text{or} \quad \operatorname{polar}(C_l)-C_l,
\]

than by the Euclidean BT-gradient. The first CF repair used a
dimension-balanced target,

\[
T_{ii}=\sqrt{d-1}(1-C_{ii}),\qquad T_{ij}=-C_{ij}.
\]

That made the trajectory monotone, but it changed the target law itself. The
more natural repair is to keep the target as unscaled \(I-C\), and change the
moment-space metric:

\[
\min_B \|W\odot(\mathcal A(B)-(I-C))\|_F^2 + \rho \|B\|_F^2,
\]

where the diagonal entries of \(W^2\) are \(d-1\), then the full matrix is
renormalized to mean weight one. This balances diagonal and off-diagonal
aggregate fitting error without changing the desired BP-like moment law.

## Result

Full CIFAR100/SimCLR, width 512, depth 12, seed 7, current-plus-random branch,
LayerNorm residual, K=4 moment batches, post-LN cap 0.25:

| Setup | Final train/test BT | Last / all-PCA acc | mean \(\|\Delta C\|\) | cos \(\Delta C,I-C\) | cos \(\Delta C,\operatorname{polar}(C)-C\) | diag norm frac |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| BP-BT greedy residual d12 | 0.0695 / n/a | n/a | 71.30 | 0.651 | 0.643 | 0.026 |
| BP-BT e2e residual d12 | 0.0795 / n/a | n/a | 79.59 | 0.654 | 0.634 | 0.035 |
| CF balanced target d12 | 0.5037 / 0.5505 | 0.1840 / 0.1888 | 6.12 | 0.427 | 0.368 | 0.158 |
| CF identity target + diag-balanced metric d12 | 0.4613 / 0.4964 | 0.1822 / 0.1906 | 6.16 | 0.275 | 0.228 | 0.131 |

The metric-weighted identity variant is a real improvement on BT and all-PCA
readout over the balanced-target variant, and all layers improve BT. But it
does not make the moment trajectory more BP-like. The layerwise \(I-C\) cosine
starts high and then collapses:

| Layer range | CF identity metric behavior |
| --- | --- |
| 1-4 | cos \(\Delta C,I-C\) stays reasonably high: 0.67, 0.62, 0.58, 0.49 |
| 5-8 | alignment decays: 0.39, 0.30, 0.25, 0.19 |
| 9-12 | late layers stop following identity flow: 0.05, -0.01, -0.10, -0.12 |

The finite/retracted update is still well predicted by the solved update
(`actual_delta_achieved_cosine = 0.913`), so this is not mainly a finite-step
or LayerNorm-retraction failure. The failure is the projected update itself:
the current branch/operator cannot realize BP's broad, low-diagonal-fraction
identity/polar moment motion at this scale.

## Interpretation

This falsifies the simple "use a better diagonal/off-diagonal metric" repair.
It lowers BT by emphasizing diagonal progress, but BP-BT's signature is the
opposite: large-norm operator motion with very small diagonal norm fraction.

The next mechanism should therefore change the reachable residual operator,
not only the target metric. The immediate object to inspect is the current
branch decomposition of \(\mathcal A(B)\): whether the current-linear block
\(HB_0\) and nonlinear block \(\phi(HA)B_1\) are failing to produce the
Sylvester-like off-diagonal flow

\[
\dot C \approx K^\top C + C K \approx I-C.
\]

Artifacts:

- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_balanced_identity_current_lawdiag_d12_b1024_ridge1e5/`
- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_identity_diagmetric_current_lawdiag_d12_b1024_ridge1e5/`
- `docs/cf_mlp_representation_learning/artifacts_smoke_identity_diagmetric_current_d2/`
