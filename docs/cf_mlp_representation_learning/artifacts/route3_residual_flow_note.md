# Route-3 Residual BT Analysis

## Question

We want to infer what kind of solution residual backprop-BT learns, and
whether its advantage is fundamentally cross-layer credit assignment or a more
local bias toward nondestructive residual refinement.

The route-3 empirical control is:

- `e2e_residual_bpbt`: saved residual BP-BT trained by backprop on the final
  layer's projector BT loss.
- `greedy_residual_bpbt`: same residual block form, but trained one layer at a
  time. Each layer optimizes a BT loss on its own output with its own
  projector; previous layers are frozen. There is no cross-layer gradient.

Both use

\[
H_{l+1}=\operatorname{LayerNorm}(H_l+\phi(H_l W_l)),
\qquad
\phi(x)=\operatorname{GELU}(x)+0.5\min(x,0).
\]

The greedy control used 100 epochs per layer. Since each greedy step trains
one block while an e2e step trains a depth-\(L\) graph, this is approximately
the same layer-forward/backward compute scale as 100 e2e epochs.

## Empirical Results

Depth-24 summary:

| Model | Hidden BT/dim | Corr diag | Shared/diff | Last linear acc | All-PCA acc | Best layer acc | Mean novelty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| e2e residual BP-BT | 0.1198 | 0.721 | 6.33 | 0.150 | 0.2166 | 0.2048 at layer 5 | 0.0094 |
| greedy residual BP-BT | 0.0297 | 0.847 | 11.33 | 0.153 | 0.2010 | 0.2292 at layer 3 | 0.0205 |

The greedy local residual model does not collapse like non-residual BP-BT and
does not collapse like aggressive agreement-expansion CF. It monotonically
improves hidden BT at all 24 layers and preserves a usable last-layer
classification readout. This is evidence against the strong claim that
multi-layer credit assignment is necessary for effective, nondestructive BT
representations in this setup.

The more precise reading is:

1. Cross-layer credit assignment is not necessary to get a strong hidden BT
   trajectory or a usable last-layer representation.
2. Cross-layer credit may still affect how useful information is distributed
   across layers: e2e has better all-layer PCA at depth 24, while greedy has a
   better early best layer.
3. The important inductive bias appears to be residual refinement plus
   normalization, not end-to-end gradient credit by itself.

The gradient-alignment diagnostic is also informative but mostly negative.
At depth 24, e2e update-vs-final-gradient cosine is only about `0.00066` on
average, and update-vs-local-gradient cosine is only about `0.0087`. Greedy
has similarly tiny update-gradient alignment. So the learned blocks should not
be understood as literal gradient steps in hidden-state space after training.
The useful object is more likely a moment/correlation trajectory than an
activation-space gradient vector field.

## What Backprop Appears To Learn

The BP-like solution is not "find the high-agreement subspace and replace the
representation." It is closer to:

\[
\text{make small residual changes that gradually move paired-view moments
toward BT while keeping the old coordinates alive.}
\]

The residual path gives nondestructiveness structurally:

\[
H_{l+1}=N(H_l+U_l),
\]

where \(U_l\) can be useful even when it is not directly interpretable as a
new representation. This is different from the non-residual CF path

\[
H_{l+1}=N(\phi(H_l A_l)),
\]

where every layer must survive a complete remapping.

The route-3 result says a CF solution should probably be greedy/layerwise, but
residual from the start. It does not need to imitate backprop's weight
gradients. It needs to imitate the statistical law of the residual moment
trajectory.

## Natural Mathematical Object

The clean object is the whitened paired-view cross-correlation operator.
Let \(Z_1,Z_2\in\mathbb R^{n\times d}\) be standardized or whitened view
representations, with approximately

\[
\frac1n Z_1^\top Z_1\approx I,\qquad
\frac1n Z_2^\top Z_2\approx I.
\]

Define

\[
C=\frac1n Z_1^\top Z_2.
\]

BT wants \(C\) near \(I\), with diagonal and off-diagonal terms weighted
differently:

\[
\mathcal L_{\operatorname{BT}}(C)
=
\|\operatorname{diag}(C)-1\|_2^2
+\lambda\|\operatorname{offdiag}(C)\|_F^2.
\]

The old hard subspace rule acts on the spectrum of a paired-difference
operator and can win by making many coordinates trivially aligned. The
residual-flow view instead asks for a small velocity \(\dot C\) that decreases
\(\mathcal L_{\operatorname{BT}}\):

\[
C_{l+1}\approx C_l+\eta \dot C_l.
\]

Preservation is not a separate instance-geometry metric. Preservation is the
bounded velocity principle:

\[
\min_{\dot C}
\langle \nabla_C \mathcal L_{\operatorname{BT}}, \dot C\rangle
+\frac{\rho}{2}\|\dot C\|_{\mathcal M}^2,
\]

subject to \(\dot C\) being realizable by a residual layer. This is natural in
the same covariance/spectral framework we are already using.

## Why The Correlation/Covariance Object

The choice of \(C\) is not meant to say "representations are only
second-order." It is a compatibility choice with the BT objective and the
existing CF parametrizer.

The existing CF-BT solver already builds layers from paired second-order
objects:

\[
\Sigma=\frac12(\Sigma_1+\Sigma_2),
\qquad
\Delta=\operatorname{Cov}(Z_1-Z_2),
\qquad
\Sigma^{-1/2}\Delta\Sigma^{-1/2}.
\]

So a residual-flow formulation should live in the same world if it is going to
be comparable. The old solver asks for a new transform that suppresses
whitened paired-difference modes. The proposed formulation instead asks for a
small residual branch whose induced movement of the BT correlation matrix
decreases the BT loss. In both cases, the fitted parameters are functions of
paired moments. The difference is not the statistics used; the difference is
the dynamical constraint:

\[
\text{old: choose a new representation map},\qquad
\text{new: choose a residual velocity}.
\]

This makes \(C\) the primary object because BT is literally a loss on \(C\),
and it makes the method comparable to CF-BT because \(C\), \(\Sigma\), and
\(\Delta\) are all second-order paired-view summaries.

This does not mean the network should be linear. The nonlinear activation is
needed to create a richer dictionary of residual velocities. The covariance
object only defines the moment-level direction we want the residual layer to
move.

## Linear Residual Flow As Derivation Only

The simplest residual map is:

\[
Z_i^+ = N(Z_i + \epsilon Z_iK),\qquad i\in\{1,2\}.
\]

Ignoring normalization terms for a first-order derivation,

\[
C^+
=
\frac1n (Z_1+\epsilon Z_1K)^\top(Z_2+\epsilon Z_2K)
\approx
C+\epsilon(K^\top C+CK).
\]

So a tied residual linear update can realize velocities of the form

\[
\dot C=K^\top C+CK.
\]

A natural closed-form update is the minimal-norm Sylvester step:

\[
K^\star
=
\arg\min_K
\left\|K^\top C+CK+\eta G\right\|_F^2
+\rho\|K\|_F^2,
\]

where

\[
G=\nabla_C\mathcal L_{\operatorname{BT}}(C).
\]

This is a direct residual covariance-flow analogue of a gradient step. It is
nondestructive because \(K\) is explicitly small, and because the layer is
identity plus a correction. No instance geometry term is needed.

This update may be too low-order because it is purely linear in the current
features. In fact, it is orthogonal to the central goal if used as the actual
deep model: stacking these updates with normalization gives a mostly linear
deep net. Its role should be derivational: it shows where the Sylvester
operator \(K^\top C+CK\) comes from and what "residual velocity in BT space"
means. It should not be the first serious implementation target.

## Nonlinear Residual Flow

The implementation target should start with a nonlinear branch:

\[
Z_i^+ = N(Z_i+\epsilon \Phi_i B),
\qquad
\Phi_i=\phi(Z_iA).
\]

Again ignoring normalization terms, the first-order cross-correlation movement
is

\[
C^+
\approx
C+\epsilon\left(
\frac1n \Phi_1^\top Z_2 B
+
\frac1n Z_1^\top \Phi_2 B
\right),
\]

up to transpose conventions depending on whether \(B\) maps branch features
back into the hidden coordinates.

This gives a known target: not unknown regression targets, but a desired
moment velocity \(-G\). We can solve

\[
B^\star
=
\arg\min_B
\left\|
\mathcal A(B)+\eta G
\right\|_F^2
+\rho\|B\|_F^2,
\]

where \(\mathcal A(B)\) is the linearized moment update induced by the branch
features. The activation becomes a dictionary of allowable covariance-flow
directions. It is not the endpoint that discards signal.

This is the most plausible route from the route-3 evidence:

1. residual from the start;
2. greedy/layerwise objective is acceptable;
3. choose a minimal-norm moment velocity that decreases BT;
4. realize that velocity through a small residual branch;
5. normalize after the residual addition.

The scale parameter should not be a cosmetic hyperparameter. We can introduce
\(\epsilon_l\) so that the total expected residual flow over depth approximates
a stable descent path:

\[
C_{l+1}\approx C_l-\epsilon_l P_l\nabla_C\mathcal L_{\operatorname{BT}}(C_l),
\qquad
\sum_{l=1}^L \epsilon_l \approx T.
\]

Here \(P_l\) is the projection from desired BT-space velocity onto velocities
realizable by the nonlinear branch dictionary at layer \(l\). A natural first
schedule is constant total flow, \(\epsilon_l=T/L\), with a trust-region check
that rejects a step if it collapses rank or worsens held-out positive-pair
BT. This is closer to "converge in totum across layers" than choosing each
layer's invariance strength independently.

## Relation To Route-2 Soft Spectral Flow

The Sylvester/residual-flow update is a more concrete version of soft spectral
flow. If \(C\) is symmetric positive enough, a geodesic move toward identity
could be written as

\[
C^+ = \exp((1-\eta)\log C),
\]

or, more generally, use the polar/SVD structure of \(C\). But directly
constructing \(C^+\) is not yet a network layer. The residual Sylvester
formulation is better because it asks: which small tied residual map produces
the desired moment movement?

So route 2 should not be "apply a spectral formula to the representation."
It should be:

\[
\text{spectral/correlation target velocity}
\quad\rightarrow\quad
\text{minimal-norm residual layer realizing it}.
\]

## Critical Evaluation

The linear Sylvester update may be too weak. It cannot create genuinely new
features, so it may improve BT only by rotating/scaling existing ones.

The nonlinear residual update may be underdetermined. If the feature
dictionary \(\Phi\) is too rich and \(\rho\) too small, it can recreate the
old destructive behavior. The trust region is therefore not a heuristic; it is
part of the mathematical object.

LayerNorm/normalization terms are not optional details. The derivation above
ignores them first-order, but the empirical residual BP-BT solution relies on
normalization after the residual addition. The next derivation should include
the projection induced by normalization, at least approximately.

The route-3 greedy result does not prove backprop never uses useful
cross-layer credit. It says that, for this setup, strong hidden BT and usable
last-layer features can be obtained without it. That is enough to justify
searching for a greedy closed-form residual rule.

## Candidate Solution To Implement First

Implement a `residual_cf_nonlinear_velocity_bt` layer:

1. Standardize paired views to \(Z_1,Z_2\).
2. Compute \(C=Z_1^\top Z_2/n\) and \(G=\nabla_C\mathcal L_{\operatorname{BT}}\).
3. Choose a nonlinear branch dictionary

\[
\Phi_i=\phi(Z_iA),
\]

   where \(A\) can initially be simple and CF-compatible: whitened random
   orthogonal directions, current CF-BT spectral directions without hard
   cutting, or a small mixture of both.
4. Solve for the minimal-norm branch readout \(B\) whose induced first-order
   moment movement best matches the desired BT velocity:

\[
B^\star
=
\arg\min_B
\|\mathcal A_A(B)+\eta G\|_F^2
+\rho\|B\|_F^2.
\]

5. Apply

\[
H_i^+ = \operatorname{Norm}(H_i+\epsilon\phi(H_iA)B^\star).
\]

The linear Sylvester layer

\[
H_i^+ = \operatorname{Norm}(H_i+\epsilon H_iK^\star)
\]

should be kept only as a diagnostic ablation: if even the nonlinear branch
does not beat it on representation quality, the branch dictionary or
linearization is wrong.

## Correction From Moment-OLS Debugging

The first implementation used a frozen-scale tangent. That was the wrong
operator for BT, because BT is a correlation objective after per-view
standardization, not an unnormalized covariance objective.

For one view, write

\[
z = \frac{x-\mu}{\sigma}.
\]

For a residual perturbation \(\Delta x\), the first-order standardized
perturbation is

\[
\Delta z
=
\Delta u
-
z\,\mathbb E[z\Delta u],
\qquad
\Delta u
=
\frac{\Delta x-\mathbb E[\Delta x]}{\sigma}.
\]

The missing term

\[
-z\,\mathbb E[z\Delta u]
\]

removes the component of the update that merely changes the view's own
per-coordinate variance. Without this projection, the fitted moment step can
have the correct sign in the unnormalized/frozen tangent but be nearly
orthogonal to the actually realized BT-correlation movement.

For branch features \(\Phi_1,\Phi_2\), update \(\Delta X_i=\Phi_iB\), and
current standardized views \(Z_1,Z_2\), the corrected linearized operator is

\[
\mathcal A(B)
=
B_1^\top M_1
+ M_2B_2
-
\operatorname{diag}(B_1^\top N_1)C
-
C\operatorname{diag}(B_2^\top N_2),
\]

where \(B_i\) means the columns of \(B\) divided by the corresponding view
standard deviations,

\[
M_1=\Phi_1^\top Z_2/n,\quad
M_2=Z_1^\top\Phi_2/n,\quad
N_1=\Phi_1^\top Z_1/n,\quad
N_2=\Phi_2^\top Z_2/n,\quad
C=Z_1^\top Z_2/n.
\]

This projected-standardization operator passed the finite-difference check:
the infinitesimal realized correlation delta matches \(\mathcal A(B)\). The
remaining limitation is therefore not the original tangent mismatch. It is
that the gradient direction reachable through the current CF-shrink branch is
still low-rank/compression-biased and much weaker than residual BP-BT.

## Stochastic Moment Estimation

The next useful correction was not a direct self-covariance penalty. Adding a
self-correlation decorrelation target to the same normal equation raised rank
only by suppressing BT descent.

Minibatch moment estimation worked better. The layer still solves a closed-form
projected moment-OLS problem, but \(C\), \(\nabla_C\mathcal L_{\rm BT}\), and
the branch moments are estimated from a per-layer minibatch and then applied to
the full representation. Empirically this improves BT and self-covariance rank
together. This suggests a mechanism closer to stepwise spectral assembly:
full-dataset moments repeatedly select dominant modes, whereas stochastic
moments perturb the projected gradient enough to assemble a broader set of
eigenmodes.

This is a plausible closed-form analogue of one useful part of SGD. It should
not be overclaimed as reproducing SGD's local-minima behavior; the mechanism
we have direct evidence for is spectral/mode diversity.

Acceptance should not be only downstream accuracy. The required diagnostics
are:

- monotone-ish hidden BT improvement across layers;
- no effective-rank collapse;
- train/test positive-pair BT generalization;
- last-layer and all-layer PCA classification readouts;
- route-3 spectral trajectory close to greedy/e2e residual BP-BT.
