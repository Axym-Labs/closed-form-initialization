# Corrected Moment-OLS BT: Mechanism And Remaining Gap

## Question

The current residual closed-form BT layer is best viewed as a projected
gradient step in moment space:

\[
H_i^+ = \operatorname{Norm}(H_i + \phi(H_i A)B).
\]

After correcting the standardization Jacobian, the fitted \(B\) realizes the
intended local correlation-space movement. The remaining question is why this
still lags residual BP-BT, and what mechanism-level changes can close the gap.

## Literature Anchors

The strongest external clue is *On the Stepwise Nature of Self-Supervised
Learning* (arXiv:2303.15438). The relevant idea for us is not the exact model,
but the framing: SSL can assemble high-dimensional embeddings one spectral
mode at a time, and kernel PCA is proposed as an analogy for SSL much like
kernel regression is an analogy for supervised learning.

NTK/lazy-training theory gives a weaker but compatible reference point:
gradient descent in wide networks can be described as kernel gradient descent
in function space (arXiv:1806.07572). Chizat et al. caution that lazy training
alone is unlikely to explain practical deep vision learning (arXiv:1812.07956),
which matters here: a pure full-dataset kernel/OLS step may be too deterministic
and too dominated by leading modes. DirectPred (arXiv:2102.06810) is another
nearby clue that useful SSL predictors/preconditioners can sometimes be set
directly from statistics rather than trained end-to-end.

## Empirical Facts So Far

The corrected projected-standardization moment operator fixed the exact
initial failure. Finite-difference/predicted correlation-delta cosine went from
`0.338` under the frozen tangent to `1.000` under the projected tangent.

Full-dataset moment OLS then gives a real monotone BT trajectory, but remains
low-rank:

- eta-32 line search, depth 24: train/test BT `0.3663/0.3822`, covariance
  effective rank `10.1`, all-PCA `0.1816`.
- eta-64 line search, depth 24: train/test BT `0.3516/0.3709`, covariance
  effective rank `10.5`, all-PCA `0.1808`.

Greedy residual BP-BT is different:

- depth 24 greedy residual BP-BT: BT `0.0297`, covariance rank about `111`
  in the stage metric, agreement soft-keep mass `197`, all-PCA `0.201`, best
  layer `0.2292`.

Thus the rank issue is not a generic property of greedy/local BT learning.
It is specific to our current deterministic CF moment direction. Non-residual
BP-BT at depth 24 does collapse to rank `1`, but it also fails BT, so that is
an architecture/optimization mismatch rather than evidence that SGD generally
causes collapse.

The direct self-covariance target was a negative result. It raised covariance
rank only by suppressing the BT step. In depth-6 calibration, self-cov weights
`0.01` and `0.1` increased rank from `10.8` to `18.5` and `20.6`, but final BT
stayed near the initial value (`0.5276` and `0.5314`). The baseline BT-only
step reduced self-correlation off-mass more than the explicit self-cov target
because it could take much larger useful steps.

The useful mechanism-level fix so far is stochastic moment estimation:

- depth-24 batch 4096: train/test BT `0.3360/0.3498`, self-cov off-mass
  `53.3`, rank `12.4`, all-PCA `0.1812`.
- depth-24 batch 1024: `0.3237/0.3374`, self-cov off-mass `28.9`, rank
  `19.6`, all-PCA `0.1930`.
- depth-24 batch 512: `0.3616/0.3750`, self-cov off-mass `24.1`, rank
  `26.6`, all-PCA `0.1892`.

Batch 1024 is the best current tradeoff. It improves BT relative to full
moments, improves rank/diversity, improves all-PCA readout, and still tracks
held-out positive pairs. Batch 512 pushes rank higher but loses BT/all-PCA,
so the effect is not "more noise is always better."

The batch-1024 mechanism also generalizes across depth on full data:

- depth 6: train/test BT `0.3920/0.3989`, rank `16.1`, all-PCA `0.1786`.
- depth 12: `0.3486/0.3603`, rank `16.4`, all-PCA `0.1848`.
- depth 24: `0.3237/0.3374`, rank `19.6`, all-PCA `0.1930`.

Batch ensembles were a diagnostic negative result. At depth 24, batch-1024
`K=2` improved train/test BT to `0.3094/0.3253`, and `K=4` gave
`0.3131/0.3278`, but the representation narrowed again: ranks `15.1` and
`12.5`, all-PCA `0.1922` and `0.1850`. Thus the stochastic benefit is not
just noise reduction. More deterministic moment agreement makes the BT step
more faithful but reintroduces low-rank mode reuse.

The current best representation-quality repair is a small linearly novel
branch component. Before solving the same projected BT-gradient moment OLS
problem, the nonlinear branch dictionary \(\Phi=\phi(HA)\) is projected against
the current representation \(H\) under the pooled train/view sample geometry:

\[
\Phi_{\perp}
=\Phi-\Pi_H\Phi.
\]

The fitted branch is then a normalized mix of \(\Phi\) and \(\Phi_{\perp}\).
This exposed the concrete failure point: after the first layer, only about
`1%` of the CF-shrink nonlinear branch energy is linearly novel
(`branch_projection_r2 ~= 0.99`). The unconstrained solver therefore mostly
selects BT-descent directions already represented by the current layer.

Pure novelty is too strong: depth-6 `mix=1.0` keeps rank high but weakens BT
to `0.5117/0.5318`. A small mix works:

- depth 12, batch 1024, `mix=0.25`: train/test BT `0.3496/0.3623`, rank
  `19.1`, last/all-PCA/best `0.1750/0.1896/0.1760`.
- depth 24, batch 1024, `mix=0.25`: `0.3328/0.3505`, rank `24.1`,
  last/all-PCA/best `0.1776/0.1956/0.1808`.

This is not an objective-level win over stochastic moment OLS, but it is a
mechanistic representation win: last-layer and all-layer readouts improve
while the BT trajectory remains monotone and train/test aligned.

Two nearby dictionary expansions were negative. Concatenating a normalized
novel block \([\Phi,\gamma\Phi_\perp]\) gives rank without useful alignment:
at depth 6, scale `0.25` reached rank `38.1` but worsened train/test BT to
`0.4404/0.4772` and all-PCA to `0.156`; larger scales raised rank above `44`
while degrading BT further. A small random-feature blend before the
nonlinearity also did not reproduce the gain: blends `0.1` and `0.25` stayed
near the old stochastic BT but did not improve rank/readout. Thus the useful
mechanism is not generic feature breadth. It is a mild residualization against
the current representation, small enough that the BT-aligned branch remains
active.

Seed and dataset checks refine the claim. On CIFAR100 SimCLR seed `8`,
`branch_novelty_mix=0.25` reproduced the seed-7 pattern at depth 12:
BT changed only from `0.3671/0.3787` to `0.3688/0.3808`, while rank improved
`15.3 -> 17.4`, last-layer readout `0.1638 -> 0.1688`, and all-PCA
`0.1788 -> 0.1832`. On Tiny ImageNet Barlow positives
(`tinyimagenet200_barlow`, 20k/5k examples), the same fix was only partial:
rank improved `39.0 -> 45.9` and last-layer readout `0.0600 -> 0.0616`, but
BT worsened `0.6087/0.6374 -> 0.6174/0.6472` and all-PCA slipped
`0.0670 -> 0.0656`.

The Tiny BP control makes this sharper. Residual BP-BT on the same Tiny
20k/5k setting reaches hidden BT/dim `0.1164` and corr diag `0.767`, but its
readout is worse than CF (`last/all-PCA/best = 0.0416/0.0594/0.0582`). Tiny
therefore separates "satisfy BT" from "learn useful class semantics" in this
MLP setup.

The new cross-correlation singular diagnostic rules out a simple coordinate
gauge explanation. Tiny CF final corr diag is `0.229`, nuclear-per-dim
`0.237`, trace/nuclear `0.966`; Tiny residual BP-BT is `0.767`, `0.768`,
`0.998`. So CF is not hiding strong paired information in a rotated basis. The
cross-view invariant signal itself is weak.

Two first attempts to repair that specific failure were negative:

- Diagonal-preconditioned BT targets (`diag_gradient_multiplier=2` or `4`)
  barely improved Tiny BT/corr diag and gave no readout gain. The target
  diagonal delta grows, but the realized diagonal delta saturates around
  `0.03-0.06` per layer under the residual branch/trust geometry.
- A `shared_cross` branch basis from symmetric cross-view covariance
  eigenvectors failed badly on Tiny: BT stayed near `0.68` and readouts fell
  below the CF-shrink branch.

A sample-space BT-gradient term is the first partial positive for invariant
signal amplification. It fits the residual branch to the activation-space
gradient of the BT objective after projecting through the standardization
tangent. On Tiny, pure sample-gradient OLS improves train/test BT
`0.6087/0.6374 -> 0.5587/0.5868`, corr diag `0.229/0.211 -> 0.294/0.272`,
and nuclear-per-dim `0.237/0.240 -> 0.300/0.291`. But it is destructive:
rank falls to `18.5`, self-covariance off-mass rises to `120.5`, and readout
falls to `0.0472/0.0496`. A moment+sample hybrid improves train BT but not
test/readout, and CIFAR pure sample-gradient is also worse than the existing
stochastic/novelty moment rules.

Adding self-covariance as a summed target does not repair sample-gradient
collapse. On Tiny, self-cov weight `0.01` with pure sample-gradient raises rank
to `125` and lowers self-cov off-mass to `4.27`, but destroys invariant signal
(`BT 0.8463/0.8757`, corr diag `0.081/0.065`). This supports a lexicographic
or constrained formulation rather than another weighted sum.

The first constrained tests refined that diagnosis. A strict self-covariance
capped line search did not change the pure sample-gradient run at all: every
full step remained feasible because self-correlation off-mass was already
decreasing (`155.8 -> 120.5`). The collapse was visible instead in covariance
effective rank (`32.7 -> 18.5`) and downstream readout (`all-PCA 0.0496`). A
rank-floor line search showed that the raw sample-gradient direction itself is
the bad object. Strict rank preservation selected essentially zero updates, and
a 2% rank-loss tolerance preserved more rank (`29.3`) but lost most of the BT
gain (`0.643/0.649`).

The working minor pivot is therefore not a scalar trust region on the raw
activation-gradient step. It is a joint moment+sample projection: keep the
moment-space BT correlation equation as the statistical/nondestructive
constraint, and add the sample-space activation gradient as an invariant-signal
amplifier. On Tiny depth 12, sample-weight `4` gives train/test BT
`0.5299/0.5843`, rank `31.4`, self-cov off `34.4`, and all-PCA `0.0608`;
sample-weight `8` gives better held-out BT `0.5356/0.5685`, rank `23.9`,
self-cov off `59.6`, and all-PCA `0.0626`. A depth check for weight `8` stays
monotone from depth 6 to 24 (`test BT 0.5993 -> 0.5663`) with rank stabilizing
near `24`, rather than decaying as in the pure sample-gradient run. This still
does not beat the moment-only Tiny readout (`0.0670` all-PCA), but it fixes the
specific destructive gradient-step failure while preserving the
objective-gradient interpretation.

Two follow-up fixes failed in informative ways. First, mode-balancing the
sample-gradient target by a smooth inverse self-covariance operator improved
rank only slightly and lost the useful held-out BT gain. This says the target is
already too old-span: preconditioning the activation displacement after it has
been formed does not create the right nonlinear modes. Second, mode-balancing
the nonlinear branch dictionary creates breadth but not invariant breadth.
Input-side branch balancing reached high effective rank and very low self-cov
off-mass but destroyed held-out BT. Output-side branch balancing slightly
improved Tiny class readout, but broke BT; adding a BT line search restored BT
and removed the readout gain. This is a useful counterexample to any simple
"rank is the bottleneck" story.

A gauge-invariant polar target is also insufficient. The tempting target is
\(UV^\top-C\), where \(C=U\Sigma V^\top\), because it asks for cross-view
singular-value growth before insisting on coordinate identity. But under the
current residual branch dictionary this target is not realized as useful
invariant signal. On Tiny, pure polar and BT+polar variants create high-rank,
low-self-covariance representations while damaging both BT and downstream
readout. Tiny BT+polar with weight `0.02` still worsens held-out BT to `0.726`,
and BT line search on weight `0.1` restores BT only by reducing the update back
toward the ordinary moment step. The failure is therefore not just a coordinate
gauge issue; it is a realizability issue.

## Mechanistic Interpretation

Compression is not the problem by itself. Good SSL should compress nuisance
variation. The bad sign is *self-covariance concentration without useful
mode assembly*: many coordinates become redundant, while downstream probes and
agreement-spectrum breadth lag behind BP-BT.

The natural framework is:

\[
\Delta C
\approx
P_{\Phi,H}\left[-\eta\nabla_C \mathcal L_{\operatorname{BT}}\right],
\]

where \(P_{\Phi,H}\) is the projection onto correlation velocities realizable
by the current nonlinear branch dictionary and normalization tangent. With full
dataset moments, this projection appears to repeatedly choose dominant
directions. Stochastic moment estimation perturbs the projected gradient, and
empirically this helps assemble more modes, closer to the stepwise SSL picture.

The novelty-projection result refines this: the CF-shrink nonlinear branch is
not giving the solver many new local directions after the first layer. The
activation is present, but most of its branch energy lies in the linear span of
the current representation. The missing ingredient is therefore not only
"better BT gradient estimation"; it is a mild, mathematically explicit bias
toward residual directions outside the currently represented linear span.

The Tiny and concat results add a second constraint: the new directions must
also be BT/semantic-aligned. Rank by itself is easy to manufacture and not
sufficient. The mechanism we need is not "increase rank"; it is "allocate
incremental modes that remain aligned with paired-view invariance and
downstream class structure."

The gauge and diagonal-preconditioning diagnostics add a third constraint:
the mechanism must amplify the magnitude of shared paired-view signal, not
merely reweight the BT objective or rotate coordinates. Backprop can raise the
correlation nuclear mass; the current CF residual dictionary mostly reduces
off-diagonal/self-covariance structure while leaving corr diag/nuclear mass
near the input level on Tiny.

The sample-gradient result adds a fourth constraint: invariant-signal
amplification is possible in the branch dictionary, but the naive activation
gradient target is mostly an old-span linear transform of the current
standardized activations. Fitting that target literally improves BT by squeezing
existing modes. The missing mathematical object is therefore a nondestructive
objective-gradient projection: the gradient signal should be represented
through moment constraints and nonlinear branch directions, not merely copied as
an activation-space displacement.

The mode-balancing results add a fifth constraint: diversity must be coupled to
invariance. Inverse-covariance operators can allocate low-energy modes, but
those modes can be semantically useful while not being view-invariant, or
view-invariant only after the line search reduces them back toward the original
moment step. The right projection should therefore optimize a joint object like
"cross-view nuclear/diagonal mass gain per unit rank loss or per unit
self-covariance concentration," not rank, covariance entropy, or target norm in
isolation.

The polar-target result adds a sixth constraint: the joint object has to live
inside the realizable branch-update cone. A mathematically natural
correlation-space target can be nearly orthogonal to what the current nonlinear
branch can realize without destroying view alignment. The next formulation
should estimate or approximate the *realized* nuclear/diagonal gain of a
candidate update, not only set a target matrix in correlation space.

The first realized-gain scan supports that framing and adds a seventh
constraint. At layer 1, candidate branch choice has weak but useful signal:
output-side mode balancing is the best one-step held-out BT/nuclear candidate,
while raw shared-cross is clearly bad. After six baseline layers, the same menu
has no positive held-out BT candidate; every nonzero branch update worsens
held-out BT, although novelty variants slightly improve one-layer readout. A
selector therefore needs a null/skip action and broader candidate generators.
If even a realized-gain selector cannot find positive invariant directions at
mid-depth, the bottleneck is the branch dictionary/update cone itself.

The broadened realized-gain scan corrects that last sentence: the mid-depth
cone is not empty, but the useful region is small. At prefix 6, scale `0.25`
for the plain or output-mode-balanced branch improves held-out BT and
cross-view nuclear mass, while full steps worsen held-out BT. Random-only and
CF+random concatenated dictionaries do not add useful invariant directions;
they mostly become near-null under scale search. A naive full-depth
validation-subset line search still fails to improve the trajectory, so the
issue is now compositional: good one-step realized gains do not automatically
compose under independent local scale selection.

The conservative schedule and rank-floor tests sharpen this further. Capping
late scales at `0.25` improves held-out BT but reduces rank and all-PCA
readout. Rank-preserving line search either stalls the objective or remains
below the baseline representation quality. Thus scalar scale choice is not the
exact missing mechanism. It can move along the BT/rank tradeoff, but it cannot
change the fact that the fitted moment-gradient direction is already a narrow
projection of the desired BT gradient.

The realized-gain selector is the cleanest cone test so far. It chooses among
multiple closed-form BT-gradient residual branches and scales at every layer.
Held-out BT selection improves Tiny test BT from `0.6374` to `0.6279`, and
cross-view nuclear selection improves it further to `0.6121` with test nuclear
mass `0.2611`. But both lose representation quality: all-PCA falls to
`0.0610` for held-out BT selection and `0.0578` for nuclear selection, versus
`0.0670` for the baseline. Nuclear selection is especially diagnostic: it
improves the invariant statistic by driving rank down to `30.2`, not by
assembling new useful modes.

Residualizing the sample-space BT-gradient target against the current
representation was the next direct test. It asks for

\[
G_\perp = (I-\Pi_H)G_{\mathrm{BT}},
\]

instead of fitting the raw activation-space gradient \(G_{\mathrm{BT}}\). This
rules out a trivial explanation: the new-mode target energy is not negligible
on Tiny (`57-58%` residual energy at depth 6). But it is not sufficient.
Fitting \(G_\perp\) improves BT relative to a no-line-search baseline, yet
all-layer readout remains below the moment baseline; adding novelty/concat
branch capacity produces breadth without alignment. Thus "new-span target" is
not the same as "useful invariant mode".

The first mechanism-level positive result is different: penalize old-span
motion in the OLS solve itself,

\[
\min_B
\|L_\Phi(B)-T_{\mathrm{BT}}\|_F^2
+\mu\|\Pi_H\Phi B\|_F^2+\rho\|B\|_F^2.
\]

This does not chase an arbitrary residual target. It keeps the BT moment
gradient as the primary object, but changes the projection so that BT descent
is biased away from residual updates linearly predictable from the current
representation. On Tiny depth 12, \(\mu=0.1\) improves held-out BT, rank,
self-covariance off-mass, and all-PCA together. On CIFAR, the same penalty
improves rank/readout but costs held-out BT. The mechanism is therefore
promising but not solved: the old-span penalty is a real nondestructiveness
bias, yet its invariant-vs-mode-creation balance is dataset-sensitive.

The next correction is to make this penalty dimensionless. A fixed
\(\mu\|\Pi_H\Phi B\|^2\) is in arbitrary operator units, while \(L_\Phi\) and
\(\Pi_H\Phi\) have dataset- and layer-dependent sensitivity. The
operator-normalized version uses a random-probe energy ratio:

\[
\min_B
\|L_\Phi(B)-T_{\mathrm{BT}}\|_F^2
+ \mu
\frac{\mathbb E_Z\|L_\Phi(Z)\|_F^2}
     {\mathbb E_Z\|\Pi_H\Phi Z\|_F^2}
\|\Pi_H\Phi B\|_F^2
+\rho\|B\|_F^2 .
\]

Empirically this is cleaner but still incomplete. On Tiny, dimensionless
\(\mu=5\) gives the best old-span result so far (`test BT 0.6273`,
all-PCA `0.0680`). On CIFAR, \(\mu=1\) is the better compromise
(`test BT 0.3622`, all-PCA `0.1904`), while \(\mu=5\) over-regularizes BT but
pushes all-PCA to `0.1912`. Thus the missing statistic is not merely operator
scale. We need a rule that adapts the invariant-vs-old-span tradeoff to the
layer's available BT gain and mode-creation risk.

The first such rule works better than either fixed scale. It solves the same
operator-normalized old-span problem over a short path
\(\mu\in\{0,1,2,5\}\), keeps candidates that preserve `95%` of the best local
train-BT gain, and chooses the smallest realized old-span update. On CIFAR100,
depth 12 matches the baseline held-out BT while improving rank/readout
(`0.3603` test BT, rank `23.7`, all-PCA `0.1934` versus baseline rank `16.4`,
all-PCA `0.1848`), and depth 24 improves the baseline on all three main
statistics (`0.3334` test BT, rank `22.2`, all-PCA `0.1952` versus
`0.3374`, `19.6`, `0.1930`). On Tiny, the rule chooses strong old-span
suppression at every layer and improves BT/rank without changing semantic
readout much: depth 24 moves `0.6428 -> 0.6330` test BT and rank
`48.6 -> 52.7`, while all-PCA stays `0.0666 -> 0.0668`.

This localizes the old-span failure more exactly. The issue was not that the
moment-gradient direction is unusable; it was that the projection step had no
local price for reusing the old representation span, and the correct price is
not a dataset-constant \(\mu\). The current adaptive rule is still crude: it is
endpoint-heavy on Tiny and uses train-BT gain rather than a held-out/nuclear
statistic. But it is the first projection-level fix that composes to depth 24
on CIFAR and improves Tiny's invariant trajectory without the rank/readout
collapse of raw sample-gradient steps.

The obvious refinements do not close the representation gap. A Pareto-knee
selector over BT gain versus old-span cost increases CIFAR rank but worsens
BT/readout; held-out positive-pair selection behaves similarly. Scoring the
path by BT plus cross-view nuclear mass is mechanistically real: CIFAR depth 24
improves to `0.3127/0.3289` train/test BT and test nuclear near `0.480`,
better than the adaptive-fraction run on invariant statistics. But all-PCA
falls to `0.1910` versus `0.1952` for adaptive fraction, and the per-layer
linear readout stops improving around mid-depth. A strict rank-preserving
line-search repair collapses the useful BT trajectory (`0.4048/0.4129`, all-PCA
`0.1734` at CIFAR depth 12). Thus rank and nuclear mass are diagnostics, not
the missing objective by themselves.

The nuanced compression picture is now sharper. Useful SSL should compress
nuisance variation, and the BT objective is partly a compression/alignment
objective. The failure is not "compression happened"; it is *misallocated
compression*: the current CF path can assemble stronger invariant mass while
discarding or failing to preserve modes that are class-useful under the linear
probe. The next mechanism has to say which modes should survive or be refined,
not merely how much rank, BT, or nuclear mass to keep.

This suggests that SGD's advantage here may not be mystical multi-layer credit
assignment. It may be partly a spectral exploration / mode-selection mechanism:
minibatch gradients do not keep selecting exactly the same global leading
moment directions.

## Current Failure Modes

1. The corrected CF rule is still far from BP-BT on BT itself:
   best current CF depth-24 BT is about `0.324/0.337`; residual BP-BT is
   `0.1198`; greedy residual BP-BT is `0.0297`.

2. Rank improves with minibatching but remains below greedy BP-BT:
   batch-1024 CF rank `19.6` versus greedy stage rank about `111`.

3. Direct self-covariance decorrelation is not a good replacement for
   stochasticity. It fights the useful BT descent instead of adding modes.

4. Generalization is only partial. CIFAR seed-8 supports the novelty-mix
   repair, but Tiny ImageNet shows that adding rank/breadth does not
   automatically improve all-layer semantic readout.

5. On Tiny, the dominant failure is low invariant-signal amplification:
   CF corr nuclear-per-dim stays near `0.237`, while residual BP-BT reaches
   `0.768`. This is not fixed by diagonal target scaling, larger trust
   regions, or a raw shared-cross eigenspace branch.

6. Sample-space BT gradients amplify invariant signal but collapse covariance
   structure and readout when fit literally. The failure is directional:
   self-cov off-mass can decrease while rank still collapses, and scalar
   rank-preserving line search gives back the BT gain. The current partial fix
   is a joint moment+sample projection, with the moment equation acting as the
   nondestructive constraint.

7. Naive mode balancing does not solve the gap. Applied to the sample target,
   it weakens invariant-signal amplification. Applied to the branch dictionary,
   it creates rank/breadth that is not BT-aligned; with line search, the breadth
   advantage disappears.

8. Gauge-invariant polar/nuclear targets do not solve the gap. They are natural
   in correlation space, but the current branch projection realizes them as
   high-rank low-alignment features rather than held-out invariant signal.

9. The first realized-gain branch scan finds only weak candidate signal at the
   first layer and no positive held-out BT candidates after six baseline
   layers. Mid-depth candidate scarcity may be the next concrete bottleneck.

10. A broadened line-search scan finds small positive mid-depth directions,
    but a full-depth validation line search does not compose into a better
    trajectory. The problem is not only candidate scarcity; it is stable
    stepwise scheduling/selection across layers.

11. A greedy realized-gain selector over candidate branches fixes part of the
    objective trajectory but still fails representation quality. The current
    update cone contains BT/nuclear-improving directions, yet these directions
    are invariant-squeezing rather than mode-assembling. The next fix must
    change the projection/dictionary, not only choose candidates or scales
    better.

12. Residualizing the sample-gradient target reveals substantial new-span
    gradient energy, but that energy is not automatically useful. This rules
    out the simple story that the raw sample-gradient failure was only
    old-span contamination.

13. Penalizing old-span update inside moment OLS is the first projection-level
    repair with the right sign. It can improve Tiny BT and all-PCA together,
    and it improves CIFAR rank/readout, but it does not fully generalize the
    objective gain. The remaining failure is not "rank" alone; it is balancing
    invariant gain against mode creation in a dataset-robust way.

14. Operator-normalizing the old-span penalty improves the formulation but does
    not remove the tradeoff. It reveals that fixed \(\mu=0.1\) corresponded to
    different dimensionless strengths across datasets. However, no single
    tested dimensionless \(\mu\) is uniformly best: Tiny prefers stronger
    old-span suppression than CIFAR if judged by joint BT/readout.

15. Adaptive old-span selection fixes the destructiveness better than fixed
    \(\mu\), but not the whole representation problem. CIFAR depth-24 now
    improves BT, rank, and all-PCA relative to the stochastic baseline. Tiny
    improves BT/rank but not class semantics, so the remaining failure is
    invariant-signal-to-semantic-mode allocation, not simply old-span reuse.

16. Local path-statistic refinements split invariant signal from useful
    representation quality. Pareto-knee and held-out-BT path selection raise
    rank but hurt readout. BT+nuclear path selection improves cross-view
    invariant statistics but still hurts all-PCA. Rank-preserving line search
    throttles the update and fails. The missing object is mode allocation, not
    another scalar proxy for compression.

## Next Mechanism Candidates

1. **Novelty-preserving stochastic moment OLS.**
   Keep batch-1024 single-batch moment estimates, but bias the nonlinear
   branch dictionary toward directions not linearly predictable from the
   current representation. The first working version is a constant
   `branch_novelty_mix=0.25`; next variants should be simple schedules or
   constraints derived from the observed branch residual energy.

2. **Mode novelty branch allocation.**
   Build \(A\) partly from underrepresented self-covariance eigenspaces or
   from residualized branch features, so each layer is biased toward new modes
   rather than reusing the current dominant ones.

3. **Lexicographic diversity constraint.**
   Instead of adding a self-covariance loss directly, solve for BT descent
   subject to a constraint that self-covariance concentration does not worsen
   too much. The failed weighted self-covariance target suggests a constraint
   or trust-region formulation is more natural than a summed objective.

4. **Stepwise spectral target.**
   Replace the full BT gradient by a schedule over BT/eigenmodes, selecting a
   controlled number of modes per layer. This is closest to the stepwise SSL
   literature and could be made closed-form if the mode-selection rule is
   derived from current correlation/self-covariance spectra.

5. **Invariant-signal amplification operator.**
   Derive a residual branch rule whose primary statistic is growth of
   cross-correlation nuclear/diagonal mass under the BT normalization tangent,
   with a secondary constraint on off-diagonal/self-covariance concentration.
   The first naive shared-cross eigenbasis failed, so this likely needs a
   bilinear or preconditioned operator rather than choosing branch directions
   directly from the current cross-covariance.

6. **Constrained activation-gradient projection.**
   Use the sample-space BT gradient as the signal-amplifying direction, but
   project or trust-region it against self-covariance concentration and rank
   collapse before fitting \(B\). The failed weighted self-cov target suggests
   a hard/lexicographic constraint: first require positive train/test BT or
   nuclear-mass gain, then choose the minimum self-covariance damage solution.

7. **Coupled nuclear-gain allocation.**
   Treat each candidate residual branch as proposing a change in cross-view
   singular/nuclear mass and a change in covariance entropy. Select or weight
   branch directions by the ratio of held-out/stochastic nuclear-mass gain to
   rank/self-cov damage. This is closer to the observed need than inverse-cov
   balancing: it asks for new modes only when they are also invariant-aligned.

8. **Realized-gain branch search.**
   Instead of choosing a target matrix first, build a small menu of candidate
   branch/update directions and score their actual one-step finite-difference
   changes in BT, cross-view nuclear mass, and covariance entropy on the
   stochastic fit batch and a held-out minibatch. This would approximate the
   realizable update cone directly and may be the closest closed-form analogue
   to SGD's mode-selection behavior.

9. **Broaden the update cone.**
   If realized-gain scoring keeps selecting null updates at mid-depth, the next
   mechanism has to generate different branch directions: multi-branch
   dictionaries, orthogonalized random nonlinear features, or residual
   predictor-like branches whose directions are not tied so tightly to the
   current CF-shrink transform.

10. **Stabilized stepwise selector.**
    Replace independent per-layer BT minimization with a conservative
    mode-assembly rule: choose small realized-gain steps only when they improve
    a smoothed held-out invariant statistic, otherwise skip or reuse the last
    successful branch family. The selector should be evaluated by whether
    gains persist over several subsequent layers, not just the current layer.

11. **Mode-creating gradient projection.**
    The selector result says the objective-gradient signal is usable but is
    projected into destructive/old-span directions. The next formulation should
    alter the fitted subspace itself: decompose the BT gradient into components
    that are predictable from the current representation and components that
    require genuinely new nonlinear residual features, then solve for the
    largest invariant gain subject to a lower bound on new-mode energy. This is
    stronger than a rank floor because the constraint would be applied before
    or during the OLS projection, not after a collapsed direction has already
    been chosen.

12. **Old-span constrained BT projection.**
    Continue from the current positive result:
    \(\|L_\Phi(B)-T_{\mathrm{BT}}\|^2+\mu\|\Pi_H\Phi B\|^2\).
    The natural next version is not just tuning \(\mu\). It should normalize
    the penalty by the realized BT gain or by branch residual energy, so that
    the method asks for "BT gain per old-span displacement" rather than an
    absolute old-span penalty whose correct scale changes across datasets and
    layers.

13. **Refine adaptive old-span tradeoff.**
    The path rule is now the current best projection-level repair. The next
    version should replace the endpoint-heavy `95%` cutoff with a smoother
    local slope,
    \(\Delta\mathrm{BT}/\Delta\|\Pi_H\Delta H\|^2\), or with a held-out
    BT/cross-view nuclear statistic. The goal is not more tuning of \(\mu\),
    but a local criterion for "invariant gain per old-span displacement" that
    remains closed-form and layerwise.

14. **Mode-allocation preserving branch/update rule.**
    The path-statistic tests suggest the next fix should alter the realizable
    update space, not only the selector. A natural target is a residual update
    that increases cross-view invariant mass only in directions that preserve
    recoverability of already useful modes, e.g. through a constrained
    low-order operator that couples BT gain to a linear-retention or
    mode-continuation condition. A scalar rank floor is too crude; the
    preservation object must be mode-specific.

Batch ensemble moment OLS should be deprioritized as a representation fix:
the K=2/K=4 tests improved BT but reduced rank/readout.
Concat-normalized novelty blocks and random-feature branch blending should
also be deprioritized: they show that branch breadth without alignment is the
wrong lever.
Diagonal-only preconditioning, larger trust regions, and raw shared-cross
branch bases should also be deprioritized for Tiny.
Naive inverse-covariance mode balancing should also be deprioritized unless it
is coupled to an explicit invariant-signal gain criterion.
Polar/nuclear targets should be deprioritized as direct moment targets; keep
them only as diagnostics or as scored statistics inside a realized-gain
selector.
