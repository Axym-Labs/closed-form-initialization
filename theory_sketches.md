# Theory Sketches for the Linear Spectral SSL Study

This note isolates the main analytic points behind the experiments in
[spectral_gap_study.py](./spectral_gap_study.py).

## 1. Pair statistics and the common-view covariance

For centered paired views `(x, x')`, define

- `Sigma = E[x x^T]`
- `Sigma' = E[x' x'^T]`
- `C = 0.5 E[x x'^T + x' x^T]`
- `Delta = E[(x - x')(x - x')^T]`
- `Sigma_bar = 0.5 (Sigma + Sigma')`

Then

`Delta = Sigma + Sigma' - E[x x'^T] - E[x' x^T]`

so

`C = Sigma_bar - 0.5 Delta`.

When the pair construction is marginally balanced, so that `Sigma' ~= Sigma`, this becomes

`C ~= Sigma - 0.5 Delta`.

This identity is the cleanest way to compare the objectives:

- PCA keeps large `Sigma`,
- shared-covariance keeps large `C`,
- hard-whitened invariance minimizes `Delta` after normalization by `Sigma_bar`,
- auto-Fisher trades `Sigma` against `Delta`,
- PCA-surplus scores principal directions using both `Sigma` and `Delta`,
- logdet-surplus scores directions by first-order gain in a logdet coding-surplus objective.

## 2. Commuting regime

The analytically easiest case is when `Sigma` and `Delta` commute:

`Sigma Delta = Delta Sigma`.

Then they share an orthonormal eigenbasis `q_i`, and

- `Sigma q_i = lambda_i q_i`
- `Delta q_i = delta_i q_i`.

In that regime, every method reduces to choosing coordinates by a scalar score.

### PCA

Choose the top `lambda_i`.

### Hard-whitened invariance

If `Sigma_bar ~= Sigma`, then the hard-whitened objective reduces to minimizing

`delta_i / lambda_i`.

This explains its pathology immediately: it can prefer tiny-variance directions so long as they are sufficiently invariant.

### Shared-covariance

Since `C ~= Sigma - 0.5 Delta`, the shared-covariance method chooses the top scores

`lambda_i - 0.5 delta_i`.

This is a first-order correction of PCA by local orbit instability.

### Auto-Fisher

The generalized eigenproblem

`Sigma v = mu (Delta + tau I) v`

reduces to choosing the top ratios

`lambda_i / (delta_i + tau)`.

This is the exact commuting-regime form of the coding-rate / nuisance tradeoff.

### PCA-surplus

The new project-specific method chooses the top scores

`log(1 + lambda_i / lambda_0) - log(1 + delta_i / delta_0)`.

So PCA-surplus is a diagonal, log-scaled coding-surplus rule on the PCA spectrum.

### Logdet-surplus

The linearized logdet-surplus rule chooses the top scores

`lambda_i / (lambda_i + sigma_0) - delta_i / (delta_i + delta_0)`.

So it is a bounded matrix-information score:

- the first term measures marginal coding gain still available in direction `i`,
- the second term measures marginal within-orbit nuisance gain spent in direction `i`.

Compared with PCA-surplus, it is less tied to the PCA basis interpretation and more naturally aligned with residual updates around the identity.

## 3. Small-Delta regime

When `Delta` is small relative to `Sigma`, all reasonable orbit-aware methods should be perturbations of PCA.

This is what happens for:

- blur-like local operators,
- smooth graph-defined barycentric positives,
- local nearest-neighbor pairings.

In that regime:

- shared-covariance is approximately PCA minus a first-order nuisance penalty,
- auto-Fisher is approximately PCA with denominator shrinkage,
- PCA-surplus is approximately PCA with additive log penalties for nuisance variation,
- logdet-surplus is approximately PCA with bounded saturation penalties in both the signal and nuisance terms.

This is exactly what the experiments show: the strong methods stay close to PCA in retained variance and probe accuracy when the within-orbit scatter is local and structured.

## 4. Why low-commutator orbits help PCA-surplus

PCA-surplus assumes that it is meaningful to evaluate nuisance variation direction-by-direction in the PCA basis.

That assumption is strongest when `Sigma` and `Delta` are close to commuting.

The study now records the normalized commutator

`||Sigma Delta - Delta Sigma||_F / (||Sigma||_F ||Delta||_F)`.

Empirically:

- single translation: about `0.183`
- random masking: about `0.161`
- blurring: about `0.080`
- block masking: about `0.075`
- graph-defined local pairs: about `0.049` to `0.054`
- channel-defined sketch pairs: about `0.056` to `0.057`

So the augmentation-free local-orbit constructions are the closest to the commuting regime.

This gives a theoretical reason why PCA-surplus is better motivated there than under stronger noncommuting perturbations like translation.

## 5. Channel-defined orbits

The graph construction is not the only augmentation-free route.

Another linear way to define a positive view is through a lossy information channel:

1. sample a sketch operator `S` with `m << p`,
2. observe the compressed measurement `y = Sx`,
3. reconstruct linearly via
   `x' = Sigma S^T (S Sigma S^T + gamma I)^{-1} y`.

Then `(x, x')` is a positive pair.

### Interpretation

This is the best linear predictor of `x` from a low-rate channel output.

So the resulting orbit notion is:

> samples are equivalent to the extent that they survive a family of lossy but generic information channels.

This has a few advantages:

- no ambient-space distance metric is required,
- no kNN graph is required,
- the construction is fully linear,
- the scaling is controlled by the sketch dimension `m`, not by pairwise search.

### Connection to the current objectives

In this setting:

- `Delta` measures what is *not recoverable* from the sketch family,
- `C` measures what is shared between the original sample and its low-rate reconstruction.

So shared-covariance, auto-Fisher, and logdet-surplus become natural spectral objectives for preserving what remains stable under generic information bottlenecks, not handcrafted transformations.

## 6. Why shared-covariance is so robust

Shared-covariance keeps doing well because it is the least ambitious correction to PCA:

- if `Delta` is small, it is just a small perturbation of PCA,
- if the pair construction is bad, it does not overreact,
- if the pair construction is good, it suppresses nuisance directions linearly.

In the current experiments, this makes it the most robust augmentation-free baseline.

## 7. Residual orbit-correction blocks

The non-residual greedy MLP experiment repeatedly applies a bottleneck:

`h^+ = U^T h`.

That means each layer discards information immediately, and later layers can never recover it. This is a poor structural match to the coding-rate viewpoint.

The residual version instead applies

`h^+ = (I + P) h`

before the fixed nonlinearity and standardization, where `P = B B^T` is a low-rank projector chosen analytically from the current orbit statistics.

### Covariance update

At the linear level this gives

`Sigma^+ = (I + P) Sigma (I + P)^T`

and

`Delta^+ = (I + P) Delta (I + P)^T`.

So the identity path preserves all previously available directions, while the low-rank branch selectively amplifies the chosen orbit-stable subspace.

### Why this matters

This is much closer to the VICReg / MCR2 / MEC intuition:

- keep existing coding volume alive,
- reduce nuisance directions through the branch choice,
- avoid repeated destructive compression.

That is exactly why the residual analytic experiments are much stronger than the plain greedy bottleneck stacks.

Empirically, with channel-defined sketch pairs:

- non-residual best on sign sketch: `0.5057`
- residual best on sign sketch: `0.8593`
- non-residual best on sparse sketch: `0.5397`
- residual best on sparse sketch: `0.8523`

So residual composition is not a minor tweak. It changes the qualitative behavior of the analytic multilayer construction.

The strongest new method in the non-residual channel-refresh stacks is `logdet_surplus`, which is consistent with its derivation as a first-order residual gain rule.

### Why it still does not beat the best shallow methods

Even with residual blocks, the branch is still only a low-rank projector chosen from local second-order orbit statistics.

That means:

- the block can refine the representation,
- but it does not create fundamentally new information,
- and it still depends on the quality of the current orbit/channel statistics.

So the residual stack can recover much of the damage caused by greedy bottlenecks, but it should not be expected to dominate the best one-shot shallow spectral solution unless the orbit construction becomes more semantically aligned.

## 8. Why hard-whitened invariance collapses

In the commuting regime, hard-whitened invariance picks directions with the smallest `delta_i / lambda_i`.

So any direction with:

- very small variance,
- but even smaller local orbit variation,

can outrank a highly informative direction.

That is not a bug in optimization. It is the correct optimum of the wrong criterion.

## 9. What would constitute a real advance

The current work suggests five increasingly ambitious directions:

1. **Robust practical baseline:** shared-covariance with channel-defined local orbits.
2. **Theory-aligned candidate:** auto-Fisher / logdet coding-rate reduction with channel-defined local orbits.
3. **Novel matrix candidate:** logdet-surplus, interpreted as a linearized coding-surplus gain rule.
4. **Novel diagonal approximation:** PCA-surplus, interpreted as a diagonal approximation to a logdet coding-surplus objective.
5. **Novel nonlinear extension:** residual orbit-correction blocks with analytic low-rank branches.

The main open theoretical task is to connect these last three items cleanly:

- derive a full logdet coding-surplus objective,
- show how logdet-surplus is its first-order residual update,
- show how PCA-surplus is the commuting / diagonal approximation of the same principle.
