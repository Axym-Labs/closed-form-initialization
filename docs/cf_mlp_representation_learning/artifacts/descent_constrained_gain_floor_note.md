# Descent-Constrained Branch Gain Floor

## Mechanism

The branch post-transform uses the paired CF gains in the whitened
paired-difference basis. A gain floor changes the gains as

\[
g'_j = g_{\min} + (1-g_{\min})g_j.
\]

The fixed-floor experiments showed the right direction but left the floor as a
global knob. The first adaptive rule chose the largest floor whose realized
full-train BT first-order term was descending:

\[
g_l = \max \{g : \langle \nabla_C L_{\mathrm{BT}}, \Delta C_l(g) \rangle < 0\}.
\]

This is mathematically too weak. On depth 24, broader candidates such as
\(g=0.75\) and \(g=1.0\) can be technically descending while providing tiny
predicted objective progress. The rule then maximizes preservation at the cost
of turning the BT update into a near-null step.

The repaired rule keeps the same preservation principle but adds a
scale-free progress constraint. For each layer, compute the realized quadratic
BT prediction

\[
\widehat{\Delta L}_g
  = \alpha_g \langle \nabla_C L_{\mathrm{BT}}, \Delta C_l(g) \rangle
    + \alpha_g^2 Q(\Delta C_l(g)),
\]

where \(\alpha_g\) is the exact clipped quadratic scale. Then choose

\[
g_l
  = \max \left\{
      g :
      \widehat{\Delta L}_g < 0
      \;\text{and}\;
      -\widehat{\Delta L}_g
      \ge
      \tau \max_{g'} \left(-\widehat{\Delta L}_{g'}\right)
    \right\}.
\]

The run below used \(\tau=0.5\). This is not meant as a tuned optimum; it is a
minimal Pareto rule: preserve as much branch breadth as possible while keeping
the BT step in the same order of local usefulness as the best candidate.

## CIFAR100/SimCLR Results

All runs use width 512, branch width 512, LayerNorm residual update,
random nonlinear branch, post-activation CF branch transform, K=4 moment
batches, LayerNorm kinetic weight 1, and exact BT-quadratic scaling.

| Setup | Depth | Train BT | Test BT | Rank | Last Acc | All-PCA Acc | Best Layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LayerNorm no floor | 12 | 0.3821 | 0.3978 | 15.5 | 0.1650 | 0.1830 | 0.1690 @ 8 |
| Fixed floor 0.15 | 12 | 0.3804 | 0.4014 | 18.0 | 0.1722 | 0.1930 | 0.1746 @ 2 |
| Descent-only adaptive, cap 0.5 | 12 | 0.4401 | 0.4846 | 43.9 | 0.1790 | 0.1990 | 0.1852 @ 5 |
| Progress-fraction adaptive | 12 | 0.3727 | 0.3995 | 24.8 | 0.1810 | 0.1916 | 0.1832 @ 8 |
| Old feature-normalized old-span adaptive | 12 | 0.3418 | 0.3603 | 23.7 | 0.1720 | 0.1934 | 0.1788 @ 6 |
| LayerNorm no floor | 24 | 0.3707 | 0.3845 | 14.2 | 0.1632 | 0.1824 | 0.1666 @ 9 |
| Fixed floor 0.25 | 24 | 0.3548 | 0.3731 | 17.0 | 0.1674 | 0.1876 | 0.1774 @ 5 |
| Descent-only adaptive, cap 0.5 | 24 | 0.3427 | 0.3792 | 30.4 | 0.1838 | 0.1948 | 0.1866 @ 21 |
| Descent-only adaptive, extended | 24 | 0.4356 | 0.5046 | 60.8 | 0.1812 | 0.1964 | 0.1928 @ 9 |
| Progress-fraction adaptive | 24 | 0.3394 | 0.3786 | 33.2 | 0.1866 | 0.1986 | 0.1922 @ 16 |
| Old feature-normalized old-span adaptive | 24 | 0.3173 | 0.3334 | 22.2 | 0.1762 | 0.1952 | 0.1768 @ 21 |

## Interpretation

The adaptive gain floor did not solve the whole representation-learning
problem, but it is the first LayerNorm residual CF-BT variant that shows
useful depth-scaled last-layer representation quality. At depth 24,
progress-fraction adaptive improves last-layer accuracy from the old
feature-normalized baseline's 0.1762 to 0.1866, and all-layer PCA from 0.1952
to 0.1986, while keeping every layer BT-improving.

The exact failure of the pure descent rule is now localized: it treats
arbitrarily small descent as enough. That lets broad branch metrics win even
when the BT step is nearly null. The progress-fraction rule fixes this by
selecting a Pareto point rather than a merely feasible point.

The remaining gap is held-out BT quality. The progress-fraction variant has
better representation readout but worse test BT than the old feature-normalized
old-span adaptive solver. This suggests the branch metric is now closer to the
right preservation/update tradeoff, but the residual update still needs a
better generalization or curvature criterion.

Artifacts:

- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_random_postcf_adaptivefloor_kinetic1_quad_k4_d12d24_b1024_ridge1e5/`
- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_random_postcf_adaptivefloor_ext_kinetic1_quad_k4_d12d24_b1024_ridge1e5/`
- `docs/cf_mlp_representation_learning/artifacts_moment_ols_cifar100_layernorm_random_postcf_adaptivefloor_prog05_ext_kinetic1_quad_k4_d12d24_b1024_ridge1e5/`
