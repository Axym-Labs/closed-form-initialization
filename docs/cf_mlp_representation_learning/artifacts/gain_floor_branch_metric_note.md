# Gain-Floor Branch Metric

## Motivation

The previous residual trust-region test localized a branch-metric failure:

\[
\text{broad nonlinear branch cone} \not\Rightarrow \text{BT descent},
\]

while strong paired CF shrink gives descent by narrowing the paired-difference
spectrum. The natural next branch metric is therefore a continuum between
paired CF shrink and whitening, not another downstream filter.

In the paired-difference whitened basis, ordinary CF shrink uses gains

\[
g_j = \frac{\lambda}{\delta_j + \lambda}.
\]

The gain-floor branch metric uses

\[
g'_j = g_{\min} + (1-g_{\min})g_j.
\]

This keeps every nonlinear mode partially alive while preserving the paired
CF bias toward low paired-difference directions. \(g_{\min}=0\) is the previous
CF-shrink branch. \(g_{\min}=1\) is whitening.

## Results

All runs below use CIFAR100/SimCLR, width 512, random nonlinear branch,
LayerNorm residual, K=4 moment batches, exact BT-quadratic scaling,
post-activation paired CF transform with invariance `1.0`, and the
LayerNorm kinetic term.

| Setup | Train BT | Test BT | Improve frac | Mean scale | Rank | Last acc | All-PCA | Best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d12, floor 0.00 | 0.3821 | 0.3978 | 1.0 | 1.0 | 15.50 | 0.1650 | 0.1830 | 0.1690 @ 8 |
| d12, floor 0.10 | 0.3790 | 0.3980 | 1.0 | 1.0 | 16.82 | 0.1664 | 0.1896 | 0.1756 @ 3 |
| d12, floor 0.15 | 0.3804 | 0.4014 | 1.0 | 1.0 | 18.04 | 0.1722 | 0.1930 | 0.1746 @ 2 |
| d12, floor 0.25 | 0.5747 | 0.5827 | 0.5 | 0.0 | 45.83 | 0.1556 | 0.1520 | 0.1556 @ 1 |
| d24, floor 0.00 | 0.3707 | 0.3845 | 1.0 | 1.0 | 14.15 | 0.1632 | 0.1824 | 0.1666 @ 9 |
| d24, floor 0.15 | 0.3609 | 0.3763 | 1.0 | 1.0 | 15.30 | 0.1612 | 0.1844 | 0.1698 @ 4 |
| d24, floor 0.25 | 0.3548 | 0.3731 | 1.0 | 1.0 | 17.02 | 0.1674 | 0.1876 | 0.1774 @ 5 |

At d12, floor `0.15` is still safely descending: every layer has negative
full-train first-order BT change (`max = -0.00629`). Floor `0.25` crosses the
boundary: every layer is vetoed (`mean scale = 0`, first-order changes all
positive).

At d24, the smaller per-layer step makes floor `0.25` feasible: all layers
have negative first-order changes and scale `1`. This means the feasible
branch breadth depends on the residual-flow step size/depth.

## Interpretation

This is a real improvement over the previous LayerNorm residual flow:

- the branch metric is changed before the OLS solve, so the residual direction
  itself changes;
- the full-train BT-quadratic guard still accepts the useful floor settings;
- all-PCA improves from `0.1830` to `0.1930` at d12 and from `0.1824` to
  `0.1876` at d24;
- last-layer readout improves at d12 (`0.1650 -> 0.1722`) and modestly at d24
  (`0.1632 -> 0.1674`).

It is not yet the desired solution. The old feature-normalized adaptive
old-span d24 result remains stronger (`test BT 0.3334`, rank `22.25`,
all-PCA `0.1952`, last `0.1762`), and greedy residual BP-BT remains much
broader. The gain floor solves part of the branch-metric problem but does not
fully prevent late spectral narrowing.

## Next Mechanism

The result supports a descent-constrained breadth rule:

\[
g_{\min,l}
= \max g
\quad \text{s.t.} \quad
\langle \nabla_C L_{\rm BT}, \Delta C_l(g)\rangle < 0.
\]

The next implementation should choose \(g_{\min,l}\) per layer by the
full-train first-order BT check, rather than using a fixed floor for all
layers and depths. This is the direct formal version of "least destructive
branch metric that still descends."
