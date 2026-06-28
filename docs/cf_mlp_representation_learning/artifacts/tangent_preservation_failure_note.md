# Tangent Preservation Failure And Gradient-Step Repair

## Question

The stable-mode preservation idea is too natural to discard after one negative
run. The precise question is whether it failed because the preservation
operator was wrong, because the selected modes were wrong, or because the
realizable branch/update cone remained too narrow.

All runs below stay in the same residual moment-OLS objective-gradient regime:

\[
H^+ = \operatorname{Norm}(H + \Phi B),
\]

with \(B\) fit by CG on linear terms. The new terms only change which linear
operator of \(\Phi B\) is penalized.

## Implementation Check

I added a standardized-tangent sample operator:

\[
L_{\mathrm{tan}}(B)
=
P_Z\left((\Phi B)S^{-1}\right),
\]

where \(P_Z(\Delta)=\Delta-\mathbb E[\Delta]-Z\,\mathbb E[Z\odot\Delta]\).
This is the first-order object that predicts realized post-standardization
motion. It can be used for stable-mode penalties and old-span penalties.

The diagnostic confirmed the raw/tangent distinction. On a smoke run, tangent
stable-mode motion had cosine `0.99` with realized post-normalization mode
drift, while raw mode motion had cosine `0.79`. On full depth-12 runs, tangent
motion remained a much better predictor of actual mode drift than raw motion.

## Controlled CIFAR100 Diagnostic

These runs used full CIFAR100 scale (`50k/5k`, width `512`, depth `12/24`,
moment batch `1024`, old-span adaptive path `0 1 2 5`). Because the local
Torch-enabled Python lacked a compatible `torchvision`, the runs used the new
PIL fallback for the SimCLR/Barlow-style augmentation path. Therefore compare
these rows internally, not as exact replacements for earlier torchvision-based
artifacts.

| Run | Train BT | Test BT | Rank | Novelty | Last | All PCA | Stable Drift | Actual-Pred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d12 baseline | 0.3582 | 0.3594 | 9.02 | 0.00097 | 0.1624 | 0.1690 | 0.0208 | 0.8697 |
| d12 raw stable | 0.3550 | 0.3567 | 9.11 | 0.00107 | 0.1584 | 0.1710 | 0.0212 | 0.8762 |
| d12 tangent stable 1 | 0.3580 | 0.3602 | 9.38 | 0.00102 | 0.1608 | 0.1702 | 0.0200 | 0.9012 |
| d12 tangent stable 5 | 0.3556 | 0.3573 | 9.10 | 0.00104 | 0.1606 | 0.1688 | 0.0204 | 0.8613 |
| d12 tangent stable 20 | 0.3589 | 0.3607 | 9.33 | 0.00104 | 0.1584 | 0.1684 | 0.0180 | 0.8912 |
| d12 tangent old-span | 0.3580 | 0.3597 | 8.69 | 0.00092 | 0.1596 | 0.1686 | 0.0206 | 0.9109 |
| d12 novelty 0.25 | 0.3528 | 0.3564 | 10.04 | 0.00138 | 0.1650 | 0.1754 | 0.0231 | 0.8598 |
| d12 novelty 0.25 + tangent old-span | 0.3530 | 0.3553 | 9.27 | 0.00133 | 0.1590 | 0.1728 | 0.0215 | 0.9091 |
| d24 baseline | 0.3538 | 0.3558 | 8.86 | 0.00065 | 0.1584 | 0.1712 | 0.0106 | 0.9513 |
| d24 novelty 0.25 | 0.3484 | 0.3514 | 9.19 | 0.00076 | 0.1648 | 0.1720 | 0.0113 | 0.9538 |

## Interpretation

The first failure point was real but not sufficient: raw stable preservation
was targeting the wrong infinitesimal object. The tangent operator fixes that
measurement and makes the local prediction faithful.

The deeper failure is mode choice. Strong tangent preservation of low
paired-difference agreement modes does reduce realized selected-mode drift
(`0.0208 -> 0.0180` at penalty `20`) but does not improve readout. Thus these
modes are not the useful preservation object.

Tangent old-span preservation also fails as a representation fix. It improves
actual-predicted local motion (`0.8697 -> 0.9109`) but narrows rank/readout.
So better local linearization fidelity is not enough.

The partial fix is branch-cone repair: mild linearly novel branch mixing
improves all-PCA and last-layer readout at both depth 12 and depth 24 while
keeping monotone BT. It is still a modest repair, not a breakthrough. The
mechanistic conclusion is:

\[
\text{current failure} =
\text{too narrow / old-span update cone}
\quad\text{more than}\quad
\text{wrong scalar objective or wrong normalization alone.}
\]

The next gradient-step variant should prioritize branch/update cone design:
new directions must be realizable after the activation and normalization
tangent, then constrained by BT and old-span preservation. Hand-selected
agreement-mode preservation is not the right central mechanism.
