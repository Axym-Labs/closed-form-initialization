# Research Memo: Closing the Gap to PCA in the Linear Hard-Whitened VICReg Setup

## Scope

This note studies the current linearized VICReg-style spectral objective in this repository and asks:

1. Why does the hard-whitened objective underperform PCA at a small bottleneck like `d = 32`?
2. What theoretically motivated changes can close that gap while staying within a linear, analytic, non-gradient framework?
3. Can the same ideas be used to optimize linear layers in a greedy MLP-style network, one layer at a time, without gradient descent?

The accompanying code for the new study is in [linear_spectral_ssl_study.py](./linear_spectral_ssl_study.py), the raw outputs are in [results/json/linear_spectral_ssl_results.json](./results/json/linear_spectral_ssl_results.json), and the cleanest derivation sketches are in [theory_sketches.md](./theory_sketches.md).

## Executive Summary

The current hard-whitened objective is not failing because of optimization. It is failing because it throws away the coding-rate / variance term that makes PCA strong.

In the linear paired-view setting, the current objective can be rewritten as a **normalized shared-covariance problem**:

- it rewards directions that are highly shared across views,
- but it normalizes away how much total variance those directions actually carry.

That means very low-variance but augmentation-invariant directions can look optimal. Empirically, that is exactly what happens: the hard-whitened method often keeps almost none of the PCA variance and therefore underperforms badly on downstream classification.

Four analytically solvable alternatives now look promising:

1. **Shared-covariance eigenspace**  
   Pick the top eigenspace of the symmetrized cross-view covariance.
   This is parameter-free and reduces exactly to PCA when there is no augmentation.

2. **Auto-Fisher / rate-reduction generalized eigenspace**  
   Solve a generalized eigenproblem of total covariance against within-orbit covariance plus an automatically chosen isotropic floor.
   This restores the missing "coding-rate vs invariance" tradeoff and also reduces to PCA in the no-augmentation limit.

3. **PCA-surplus / orbit-aware principal selection**  
   First diagonalize the total covariance as PCA does, then rank principal directions by a coding-surplus score that rewards variance and penalizes local orbit variation.
   This is more project-specific and more novel than simply switching to another generalized-eigen method.

4. **Logdet-surplus / linearized residual coding gain**  
   Linearize a coding-rate-minus-within-orbit logdet objective around the identity map and keep the directions with the largest first-order gain.
   This keeps the method linear and spectral, but is especially natural for residual analytic blocks.

Empirically, all four variants are much closer to PCA than the current hard-whitened objective, and the most interesting current candidates are:

- `auto_fisher` as the strongest MCR2/MEC-aligned method,
- `logdet_surplus` as the strongest novel channel-defined multilayer direction,
- `pca_surplus` as the clean diagonal approximation with the simplest interpretation.

One further empirical point now seems clear: improving the local orbit construction helps a little, but the main bottleneck was the objective, not the exact orbit recipe. Once the objective is fixed, even simple augmentation-free pair constructions are already near PCA.

I also added augmentation-free pair constructions of two kinds:

- graph-defined local positives,
- channel-defined positives obtained by reconstructing a sample from random low-dimensional sketches.

In both settings, shared-covariance and auto-Fisher match or nearly match PCA, which is encouraging for the "stop hand-designing augmentations" objective.

## 1. What the Current Objective Is Really Doing

The current setup defines

- `Sigma_x = E[x x^T]`
- `Delta_avg = E[(I - A_k) Sigma_x (I - A_k)^T]`
- `Sigma_bar = E[0.5 (Sigma_x + A_k Sigma_x A_k^T)]`

and solves

`min tr(Y N Y^T)` subject to `Y Y^T = I_d`,

with

`N = Sigma_bar^{-1/2} Delta_avg Sigma_bar^{-1/2}`.

The encoder is

`W = Y Sigma_bar^{-1/2}`.

### Key identity

Define the symmetrized cross-view covariance

`C_avg = E[0.5 (Sigma_x A_k^T + A_k Sigma_x)]`.

Then for every augmentation,

`Delta_k = Sigma_x + A_k Sigma_x A_k^T - Sigma_x A_k^T - A_k Sigma_x`

and therefore

`C_avg = Sigma_bar - 0.5 Delta_avg`.

So the current problem is equivalent to the generalized eigenproblem

`max tr(U^T C_avg U)` subject to `U^T Sigma_bar U = I_d`.

This is the crucial point: the current method is **not** maximizing shared covariance directly. It is maximizing **shared covariance normalized by total view covariance**.

That is a correlation-like criterion, not a coding-rate criterion.

## 2. Why PCA Wins

PCA solves

`max tr(U^T Sigma_x U)` subject to `U^T U = I_d`.

So PCA selects the `d`-dimensional subspace with the largest retained variance.

The hard-whitened objective instead selects the subspace with the smallest normalized augmentation sensitivity. These are not the same objective, and in fact they diverge sharply when:

1. some directions are very invariant but carry little variance,
2. some label-relevant directions are moderately variant under augmentation,
3. the augmentation family does not align exactly with the nuisance subspace.

On MNIST, all three happen.

### No-augmentation degeneracy

If `A_k = I`, then `Delta_avg = 0`, hence `N = 0`, and every whitened subspace is optimal.

This is the clearest sign that the objective has lost the variance / entropy / coding term. PCA does not degenerate in this case; it becomes the natural optimum.

### Coding-rate perspective

This is exactly the perspective emphasized by Yi Ma and the MCR2/MEC line of work:

- a good representation should suppress nuisance variability,
- but it must still preserve substantial coding rate / entropy / volume in the useful directions.

Hard whitening makes the "spread" term constant, so the current objective keeps only the nuisance-compression part.

That is why it can be beautifully optimizable and still produce poor features.

## 3. A Better Linear Objective: Shared Covariance

The simplest fix is to stop normalizing shared covariance away.

### Objective

Use

`max tr(U^T C_avg U)` subject to `U^T U = I_d`.

The solution is just the top-`d` eigenspace of `C_avg`.

### Why this is principled

- It is still fully linear and spectral.
- It is still tied to paired-view agreement.
- It restores the missing preference for directions that are both invariant **and** high-energy.
- If `A_k = I`, then `C_avg = Sigma_x`, so it **exactly becomes PCA**.

So this objective preserves the original motivation while eliminating the no-augmentation pathology.

### Interpretation

This is the common-variance subspace of the two views. It is the natural linear object if one wants to retain what is shared across views rather than merely what changes least after whitening.

## 4. A More Novel Fix: PCA-Surplus

The previous fixes still sit fairly close to known generalized-eigen viewpoints. A more novel idea within this project is to keep the representation anchored to PCA directions and use invariance only as a **spectral reweighting of the principal spectrum**.

### Construction

Let

`Sigma_x = Q diag(lambda_1, ..., lambda_p) Q^T`

be the PCA eigendecomposition. For each PCA direction `q_i`, define its orbit variation

`delta_i = q_i^T Delta q_i`.

Now score each principal direction by

`s_i = log(1 + lambda_i / lambda_0) - log(1 + delta_i / delta_0)`,

where

- `lambda_0 = tr(Sigma_x) / p`
- `delta_0 = tr(Delta) / p`

and choose the top-`d` PCA directions by `s_i`.

### Why this is attractive

- it is still linear and analytic,
- it is parameter-free in the current implementation,
- it reduces exactly to PCA when `Delta = 0`,
- and it prevents the low-energy rotation pathology of the hard-whitened objective because it can only choose among actual principal coding directions.

### Coding-rate interpretation

This can be read as a diagonal approximation to a coding-surplus objective:

- the first term measures marginal coding contribution,
- the second term measures marginal nuisance coding spent within the local orbit,
- the selected subspace maximizes net coding surplus over the PCA spectrum.

This is the method called `pca_surplus` in the new study.

### A residual-friendly variant: logdet-surplus

The PCA-surplus rule is diagonal in the PCA basis. A slightly more ambitious but still fully analytic alternative is to start from the matrix objective

`J(T) = logdet(T Sigma_x T^T + sigma_0 I) - logdet(T Delta_avg T^T + delta_0 I)`

and linearize it around the identity map `T = I + eta P`.

To first order in `eta`, the gain is

`dJ/deta |_(eta=0) = 2 tr(P G)`,

with

`G = Sigma_x (Sigma_x + sigma_0 I)^(-1) - Delta_avg (Delta_avg + delta_0 I)^(-1)`.

So the best rank-`d` symmetric correction is the top-`d` eigenspace of `G`.

This is the method called `logdet_surplus` in the new study. It is attractive because:

- it is still linear and spectral,
- it has a direct coding-rate interpretation,
- it is naturally aligned with residual blocks,
- and it avoids the hard-whitened tendency to rotate into tiny-variance directions.

## 5. A Stronger Fix: Auto-Fisher / Rate-Reduction Generalized Eigenproblem

The next step is to reintroduce the within-orbit vs total-coding tradeoff more explicitly.

Treat `Delta_avg` as a within-orbit scatter matrix and `Sigma_x` as the total scatter. Then solve

`Sigma_x v = lambda (Delta_avg + tau I) v`.

Take the top generalized eigenvectors.

### Why this makes sense

This is a self-supervised analog of Fisher/LDA-style subspace selection:

- numerator: keep directions with high total information content,
- denominator: penalize directions that fluctuate strongly inside the augmentation orbit.

It is also the finite-dimensional linear surrogate of the MCR2/MEC viewpoint:

- maximize representation coding rate,
- minimize within-orbit coding / nuisance variation.

### Automatic floor

To avoid the no-augmentation degeneracy without introducing a tuned hyperparameter, I used

`tau = tr(Delta_avg)/p + 1e-6 * tr(Sigma_x)/p`.

This is not meant as a final theorem, but it is a clean automatic floor:

- if augmentations are strong, the denominator reflects their empirical energy,
- if augmentations vanish, the denominator becomes approximately isotropic,
- in that limit the method reduces back to PCA.

This variant is called `auto_fisher` in the new script.

## 6. What the New Experiments Show

### 6.1 Shallow spectral comparison at `d = 32`

The shallow study compares:

- `pca`
- `pca_surplus`
- `logdet_surplus`
- `hard_whitened_invariance`
- `shared_covariance`
- `auto_fisher`

across four suites:

- `single-translation`
- `blurring`
- `random-masking`
- `block-masking`

#### Main result

The current hard-whitened objective is the only method that is consistently poor.

It often retains almost none of the PCA variance:

- single translation: retained variance is about `0.0015` of PCA
- blurring: retained variance is about `0.00003` of PCA
- random masking: retained variance is about `0.00027` of PCA
- block masking: effectively zero

This is exactly the predicted failure mode.

#### Probe accuracies

| suite | PCA | PCA-surplus | logdet-surplus | hard-whitened | shared-cov | auto-Fisher |
|---|---:|---:|---:|---:|---:|---:|
| single-translation | 0.9060 | 0.8120 | 0.6897 | 0.4143 | 0.8467 | 0.8217 |
| blurring | 0.9060 | 0.9060 | 0.9060 | 0.6600 | 0.9067 | 0.9207 |
| random-masking | 0.9060 | 0.9033 | 0.8527 | 0.6677 | 0.9040 | 0.8903 |
| block-masking | 0.9060 | 0.8177 | 0.7387 | 0.1147 | 0.9060 | 0.8620 |

### 6.2 Augmentation-free graph pairs

I also tested a local-pair variant with no handcrafted image augmentations at all:

- for each sample, either pair it with its 1-nearest neighbor in the training set,
- or pair it with the mean of its 5 nearest neighbors,
- build the same spectral statistics from those data-defined pairs,
- solve the same linear spectral problems.

#### Probe accuracies

| graph-pair setting | PCA | PCA-surplus | logdet-surplus | hard-whitened | shared-cov | auto-Fisher |
|---|---:|---:|---:|---:|---:|---:|
| 1-nearest-neighbor positives | 0.9063 | 0.8727 | 0.9070 | 0.1147 | 0.9067 | 0.9073 |
| 5-nearest-neighbor mean view | 0.9057 | 0.8873 | 0.8903 | 0.1147 | 0.9053 | 0.9003 |
| mutual-10NN positives | 0.9063 | 0.8640 | 0.9050 | 0.1147 | 0.9077 | 0.9073 |
| affinity-10NN mean view | 0.9063 | 0.8877 | 0.8913 | 0.1147 | 0.9060 | 0.8997 |

#### Commutator diagnostic

The study now also records

`||Sigma Delta - Delta Sigma||_F / (||Sigma||_F ||Delta||_F)`,

which measures how close the pair statistics are to the commuting regime in which PCA-surplus is most naturally justified.

For the graph-defined pairs, the commutator ratios are low:

- 1NN graph: about `0.054`
- 5NN mean graph: about `0.050`
- mutual-10NN graph: about `0.054`
- affinity-10NN mean graph: about `0.049`

### Interpretation

This is important for the secondary objective.

It suggests the real issue is not "we need more clever handcrafted augmentations." The deeper issue is that the hard-whitened criterion is wrong for the job. Once the objective is repaired, simple data-defined local-pair constructions already become competitive with PCA.

1. `shared_covariance` nearly closes the gap to PCA without any extra tuning.
2. `auto_fisher` can exceed PCA when the orbit construction aligns well enough with nuisance structure, especially for blur.
3. `logdet_surplus` is the stronger new matrix-information candidate on augmentation-free graph pairs, even though it does not beat the best shallow baseline.
4. `pca_surplus` remains the clean diagonal approximation, but it is not the strongest new method overall.
5. The newer graph constructions do not dramatically outperform the simple 1NN / 5NN variants. This suggests the objective matters more than sophisticated local-orbit engineering, at least at the current level.
6. The current hard-whitened method badly over-prioritizes invariance over information retention.

### 6.3 Metric-free channel pairs

To avoid any reliance on ambient-space distance metrics, I also tested a second augmentation-free construction:

- draw a random low-dimensional sketch operator `S`,
- observe only the sketch `Sx`,
- form the best linear reconstruction
  `x' = Sigma S^T (S Sigma S^T + gamma I)^{-1} S x`,
- treat `(x, x')` as a positive pair.

This is a self-conditioned, channel-defined view rather than a neighborhood graph. It is generic, analytic, and scalable because the sketch dimension can be kept small and the sketch itself can be sparse.

#### Probe accuracies

| channel-pair setting | PCA | PCA-surplus | logdet-surplus | hard-whitened | shared-cov | auto-Fisher |
|---|---:|---:|---:|---:|---:|---:|
| sign-64 sketch | 0.9060 | 0.8857 | 0.8877 | 0.1147 | 0.9063 | 0.8957 |
| sparse-64 sketch | 0.9060 | 0.8763 | 0.8913 | 0.1147 | 0.9030 | 0.9017 |

#### Interpretation

This matters because it avoids the main objection to graph-based local orbits:

- there is no kNN search,
- there is no dependence on high-dimensional ambient distance,
- the pair construction is distribution-adaptive through `Sigma`,
- and the computations naturally scale through low-dimensional sketches.

The empirical pattern is consistent with the graph-based findings:

1. the hard-whitened objective still collapses,
2. shared-covariance remains extremely robust,
3. auto-Fisher remains competitive,
4. logdet-surplus is the strongest novel method in the channel-defined shallow setting, though still slightly below the best classical baselines,
5. PCA-surplus remains a useful diagonal approximation, but it is no longer the strongest new method.

## 7. Layerwise Analytic MLP Experiment

To test whether these linear spectral objectives can optimize linear layers in a network one layer at a time, I implemented a greedy pipeline:

1. solve a spectral problem for the current layer,
2. freeze the resulting linear map,
3. apply a fixed ReLU and standardization,
4. repeat for the next layer.

No gradient descent is used.

The layer widths were `[256, 64, 32]`.

This is not a proof of anything yet, but it is a clean analytic testbed for "linear-layer optimization inside a DNN-like stack."

### Results

#### Single-translation

| method | final probe accuracy |
|---|---:|
| logdet-surplus | 0.6947 |
| auto-Fisher | 0.6197 |
| shared-covariance | 0.4797 |
| PCA | 0.4780 |
| PCA-surplus | 0.4017 |
| hard-whitened invariance | 0.3503 |

#### Blurring

| method | final probe accuracy |
|---|---:|
| logdet-surplus | 0.7863 |
| PCA-surplus | 0.6277 |
| auto-Fisher | 0.5640 |
| hard-whitened invariance | 0.5167 |
| shared-covariance | 0.5050 |
| PCA | 0.4150 |

#### Graph-refresh layerwise training

| pair construction | best method | final probe accuracy |
|---|---|---:|
| 1NN graph refresh | shared-covariance | 0.5433 |
| 5NN mean graph refresh | shared-covariance | 0.5363 |
| mutual-10NN graph refresh | shared-covariance | 0.5220 |
| affinity-10NN mean graph refresh | shared-covariance | 0.5390 |

#### Channel-refresh layerwise training

| pair construction | best method | final probe accuracy |
|---|---|---:|
| sign-64 sketch refresh | logdet-surplus | 0.6260 |
| sparse-64 sketch refresh | logdet-surplus | 0.5577 |

#### Channel-residual analytic training

I also tested a residual version of the layerwise construction. Instead of replacing the representation by a new bottleneck at each step, each block applies an identity plus a low-rank orbit-aware correction:

`h_{l+1} = relu((I + P_l) h_l)`,

where `P_l` is the rank-128 projector selected analytically from the current channel-defined pair statistics. After three such residual blocks, a final analytic projection to dimension `32` is learned.

This is still fully analytic and still uses no gradient descent, but it is much closer to the coding-rate intuition because the identity path keeps previously available information alive.

| residual channel setting | PCA | PCA-surplus | logdet-surplus | hard-whitened | shared-cov | auto-Fisher |
|---|---:|---:|---:|---:|---:|---:|
| sign-64 sketch residual | 0.8523 | 0.8303 | 0.8530 | 0.1147 | 0.8513 | 0.8593 |
| sparse-64 sketch residual | 0.8523 | 0.8153 | 0.8513 | 0.1147 | 0.8447 | 0.8483 |

### Interpretation

The main empirical sign in the layerwise study is:

- once the layerwise problem becomes genuinely compositional,
- a pure variance criterion like PCA is no longer obviously the best local rule,
- while an orbit-aware information-vs-invariance rule can do better.

So for greedy layerwise analytic training, the right comparison is no longer "can we exactly match shallow PCA?" but rather:

> can we use augmentation-aware linear spectral rules to build better deep greedy networks than greedy PCA?

The updated evidence now separates two different stories.

1. Plain greedy bottleneck stacks are too destructive. They repeatedly discard coding dimensions, so even when the local rule is sensible they fall far below the best shallow spectral methods.
2. Residual analytic stacks are much better behaved. On the channel-defined runs, the best residual model jumps from the `0.50`-range to the `0.85`-range, which is a large gain over the non-residual greedy setup.

This matters conceptually. The residual construction is not just a better engineering choice. It is closer to the VICReg / MCR2 / MEC viewpoint because it preserves existing coding rate through the identity path while only adding a low-rank orbit-aware correction.

At the same time, the residual stacks still do **not** beat the best shallow channel-based spectral baselines, which remain around `0.90` to `0.91`. So the multilayer result is now more interesting than before, but still secondary to the shallow theory.

The updated nonlinear picture is now:

- `logdet_surplus` is the strongest novel method in the plain channel-refresh multilayer setting,
- `auto_fisher` remains strongest on the sign-sketch residual run,
- PCA is still a very strong residual baseline once the identity path prevents repeated information loss.

At the same time, this is where caution is needed. These greedy MLP-style constructions are still underperforming the best shallow baselines. So the multilayer setting is currently more valuable as a **theoretical testbed** than as a final practical recipe.

The narrower conclusion is:

- the shallow hard-whitened criterion is fundamentally the wrong objective,
- shallow channel-defined orbit methods already solve much of the augmentation problem without any metric search,
- graph-defined shallow orbit methods are useful sanity checks but not the preferred scalable story,
- residual analytic channel stacks are far better than plain greedy bottleneck stacks,
- `logdet_surplus` is currently the strongest novel multilayer/channel method,
- hard-whitened invariance still fails even in the residual setting,
- greedy layerwise analytic training is interesting, but it is not yet strong enough to be the central claim.

## 8. How This Connects to VICReg, MCR2, MEC, and Yi Ma's Perspective

### VICReg

VICReg combines:

- invariance,
- variance,
- covariance regularization.

The current hard-whitened setup kept only the analytically convenient invariant core after hard whitening. That made the problem clean, but it also removed the part of the objective that prevents the solution from drifting to low-information directions.

### MCR2

MCR2 says a good representation should:

- preserve large overall coding rate,
- reduce coding rate within classes / groups / orbits.

In the self-supervised paired-view setting, the augmentation orbit plays the role of a pseudo-class. Then:

- `Sigma_x` is total coding structure,
- `Delta_avg` is within-orbit variation.

The auto-Fisher objective is the linear generalized-eigen version of this idea.

### MEC

MEC emphasizes maximum entropy among plausible representations. In this linear setting, that means you should not only keep invariant directions, but also preserve directions with substantial spread / coding volume.

Again, the current hard-whitened objective loses exactly this piece, while shared-covariance and auto-Fisher restore it.

### Yi Ma / coding-rate viewpoint

The clearest conceptual summary in that language is:

> The current objective compresses nuisance variation, but because of hard whitening it no longer rewards preserving useful coding rate. To close the gap to PCA, we need a linear spectral objective that simultaneously preserves total coding and suppresses within-orbit coding.

That is precisely what the shared-covariance, auto-Fisher, PCA-surplus, and logdet-surplus objectives are trying to do, in slightly different ways.

## 9. How to Reduce Manual Augmentation Engineering

This was the weaker point of the current setup. The most principled next move is not "better handcrafted image operators" but **augmentation-free orbit structure**, preferably through scalable channel-defined views.

The clean linear routes are:

1. define a self-conditioned channel view through a low-dimensional sketch and linear MMSE reconstruction,
2. optionally build a positive-pair graph from the data itself as a secondary local-orbit check,
3. form the corresponding within-pair scatter,
4. solve a linear spectral objective such as shared-covariance, auto-Fisher, PCA-surplus, or logdet-surplus with that pair structure.

This connects directly to:

- Slow Feature Analysis (slowness / invariance),
- Locality Preserving Projections (graph-based spectral dimensionality reduction),
- graph-Laplacian / local-orbit viewpoints.

In other words, augmentations can be replaced by a **low-rate information channel**, and secondarily by a **local equivalence graph**. The former is the cleaner long-term direction here because it avoids high-dimensional metric search and scales through sketching machinery rather than neighbor search.

I implemented simple versions of both ideas in the new study:

- a 1-nearest-neighbor positive-pair graph,
- a 5-nearest-neighbor mean positive view.
- a dense random sign sketch channel,
- a sparse random sketch channel.

Even these plain constructions are already competitive with PCA once the objective is changed from hard-whitened invariance to a coding-aware alternative such as shared-covariance, auto-Fisher, or logdet-surplus.

## 10. Recommended Direction

If the goal is to stay close to the current motivation and keep everything linear / analytic, the strongest path is:

1. **Retire the pure hard-whitened objective as the main method.**  
   It is analytically elegant but structurally biased toward low-variance invariant directions.

2. **Promote shared-covariance as the zero-tuning baseline.**  
   It is the cleanest linear fix and recovers PCA when augmentations disappear.

3. **Promote auto-Fisher as the strongest theory-aligned candidate.**  
   It is still analytic, still linear, closer to MCR2/MEC, and already gives the strongest empirical evidence in this repository.

4. **Promote logdet-surplus as the strongest novel candidate.**  
   It is project-specific, analytic, naturally matched to residual blocks, and is currently the best new method in the multilayer channel-defined setting.

5. **Keep PCA-surplus as the diagonal approximation.**  
   It remains useful because it is simple, parameter-free, and easy to analyze, even if logdet-surplus is now empirically stronger.

6. **Move toward augmentation-free pair constructions instead of manual augmentations.**  
   This now has preliminary empirical support in the current repository, and the channel-defined self-conditioned view is the stronger long-term direction.

7. **Use residual analytic stacks, not plain greedy bottleneck stacks, for the nonlinear extension.**  
   The identity path preserves coding rate and makes the multilayer experiment materially stronger without leaving the analytic regime.

8. **Keep graph-defined local orbits as a diagnostic, not the main story.**  
   They are useful for theory checks and commuting-regime evidence, but they are not the preferred scalable formulation.

## 11. Concrete Next Steps

1. Strengthen the augmentation-free pair construction:
   - structured sparse sketch channels
   - multiscale reconstruction channels
   - low-rank / fast Johnson-Lindenstrauss sketch families
   - channel families that mimic generic missing-information processes rather than image-specific operators

2. Theorize logdet-surplus more cleanly:
   - derive the linearized residual rule directly from a global logdet coding-surplus objective
   - characterize its commuting-regime scalar score and compare it to auto-Fisher
   - understand when it should outperform the diagonal PCA-surplus approximation

3. Theorize PCA-surplus more cleanly:
   - derive it as a diagonal approximation to a logdet coding-surplus objective
   - characterize when it dominates PCA and when it reduces exactly to PCA
   - compare its selection rule to generalized-eigen auto-Fisher

4. Theorize residual analytic blocks:
   - model `h^+ = (I + P) h` as a low-rank coding-rate correction rather than a bottleneck replacement
   - characterize when repeated residual orbit corrections can improve on one-shot shallow projections
   - explain why hard-whitened blocks still collapse even with an identity path

5. Replace the current reporting with:
   - coding-rate proxies,
   - retained variance,
   - within-orbit scatter,
   - pairwise shared covariance

6. Derive the full logdet rate-reduction objective:
   - `max logdet(W Sigma_x W^T + eps I) - logdet(W Delta_avg W^T + eps I)`
   - compare its one-shot optimum to auto-Fisher
   - compare its linearized residual update to logdet-surplus

7. Study generative conditions under which the hard-whitened method can beat PCA:
   - signal-nuisance decomposition,
   - augmentation acts only on nuisance,
   - nuisance variance dominates total covariance

## 12. Closed-Form Barlow Twins Layer

The newest analytic DNN layer in [closed_form_barlow_twins.py](./closed_form_barlow_twins.py) simplifies the earlier Sylvester model to a one-parameter objective.

Let

- `M = Sigma_bar^{-1/2} Delta Sigma_bar^{-1/2}`

and solve in whitened coordinates

`min_G tr(G^T M G) + lambda ||G - I||_F^2`.

This is the cleanest quadratic approximation we currently have to a Barlow-style "keep covariance near identity while suppressing view disagreement" objective.

### Closed form

Because `M` is symmetric positive semidefinite, the objective is strictly convex for every `lambda > 0`, and its unique minimizer is

`G* = lambda (M + lambda I)^(-1)`.

If `M = Q diag(mu_i) Q^T`, then

`G* = Q diag(lambda / (mu_i + lambda)) Q^T`.

So the layer acts as a spectral shrinkage filter on the whitened disagreement modes:

- directions with large disagreement `mu_i` are shrunk strongly,
- directions with small disagreement are preserved,
- the gain always lies in `(0, 1]`.

This lets us fix `lambda_inv = 1` without loss of generality and expose a single remaining scalar `lambda`.

### Why the old `lambda_shared` became unnecessary here

In whitened coordinates,

- `S = Sigma_bar^{-1/2} C Sigma_bar^{-1/2}`
- `M = Sigma_bar^{-1/2} Delta Sigma_bar^{-1/2}`

and since `Delta = 2 (Sigma_bar - C)`, we have

`M = 2 (I - S)`.

So the invariance matrix and the whitened shared-covariance matrix have the same eigenvectors. In this approximation, `lambda_shared` does not introduce a new geometry; it only changes the scalar gain curve on the same modes. Empirically, removing it slightly improved the current MNIST runs, so the model is now parameterized only by `lambda`.

### Current empirical status

The earlier bottom-`r` stable-subspace split was removed because it was not helping the full-width representation and only weakly affected the final compressed readout.

At depth `3` with `lambda = 1` in the current full-space model:

- `single-translation`: full-width probe `0.9450`, final PCA-32 probe `0.8763`
- `block-masking`: full-width probe `0.9523`, final PCA-32 probe `0.9043`

So this one-parameter layer is materially stronger than the earlier `preserve + identity` surrogate, even though it is still a second-order approximation.

## 13. Computational Complexity and Convergence

The right comparison is not only "how many flops" but also "what numerical problem is being solved." The current implementation is a full-space layer-shaping solver, not a deflated residual update.

All of the analytic methods here are exact dense linear-algebra solvers. The real differences are:

- how many covariance-like matrices must be built,
- how many dense spectral decompositions are needed,
- how well conditioned the resulting solve is,
- whether the objective is unique or gap-sensitive.

### Asymptotic comparison

Let `n` be the sample count, `p` the ambient width, and `d` the bottleneck used only for the final probe.

| method | exact solve | asymptotic cost |
| --- | --- | --- |
| PCA | top eigenspace of `Sigma` | `O(n p^2 + p^3)` |
| whitened PCA | PCA plus eigenvalue rescaling | `O(n p^2 + p^3)` |
| hard-whitened invariance | whiten `Sigma_bar`, then bottom eigenspace of `Sigma_bar^{-1/2} Delta Sigma_bar^{-1/2}` | `O(K n p^2 + 2 p^3)` |
| auto-Fisher | generalized eigenspace of `(Sigma, Delta + tau I)` | `O(K n p^2 + p^3)` |
| closed-form Barlow Twins layer | whiten `Sigma_bar`, form `M = Sigma_bar^{-1/2} Delta Sigma_bar^{-1/2}`, then apply `G* = lambda (M + lambda I)^(-1)` | `O(n p^2 + p^3)` per layer |

The important update is that, after removing the old stable-split / bottom-`N` stage, the closed-form Barlow Twins layer is now in the **same asymptotic complexity class** as PCA and auto-Fisher:

- all three are `O(n p^2 + p^3)` dense solvers,
- but the one-parameter layer has a much larger cubic constant because it uses multiple dense matrix products and two eigendecompositions rather than one.

### Practical timing on MNIST

Using [analytic_complexity_compare.py](./analytic_complexity_compare.py) on `MNIST`, `n = 12000`, `p = 784`, `d = 32`, `suite = single-translation`, `lambda = 1`:

| method | moment build (s) | analytic solve (s) | key numerical feature |
| --- | --- | --- | --- |
| PCA | `0.126` | `0.061` | top-`d` eigengap `2.87e-2` |
| whitened PCA | `0.126` | `0.064` | same eigengap as PCA |
| hard-whitened invariance | `0.963` | `0.185` | bottom-gap `4.61e-17`, nearly degenerate |
| auto-Fisher | `0.961` | `0.093` | denominator cond. `84.6` |
| closed-form Barlow Twins layer | `0.930` | `0.253` | solve cond. `4.08` for `M + lambda I` |

These timings were rerun with single-threaded BLAS to reduce noise. They are still implementation-specific, but they now match the asymptotic story much more closely.

The most useful operational breakdown is for the one-parameter layer itself:

- pair statistics: `0.930s`
- eigendecompose `Sigma_bar`: `0.063s`
- build `Sigma_bar^{1/2}` and `Sigma_bar^{-1/2}`: `0.035s`
- form `M = Sigma_bar^{-1/2} Delta Sigma_bar^{-1/2}`: `0.034s`
- eigendecompose `M`: `0.064s`
- reconstruct `T = Sigma_bar^{1/2} G Sigma_bar^{-1/2}`: `0.051s`

So the extra cost is no longer a separate `N`-based stable-subspace computation. It is simply the cost of doing **more dense linear algebra than PCA** on the same `p x p` scale.

### Convergence / numerical behavior

The main convergence story is now quite clean:

1. **PCA / whitened PCA**  
   Both are one-shot eigendecomposition problems. They reach the global optimum exactly, and subspace stability is governed by the top eigengap.

2. **Hard-whitened invariance**  
   This is also a one-shot spectral solve, but it is much more fragile numerically because it depends on whitening `Sigma_bar` and then selecting the bottom eigenspace of a matrix that often has a tiny eigengap. In the current `single-translation` run, that gap is essentially zero.

3. **Auto-Fisher**  
   This is a one-shot generalized eigenproblem. Once the denominator is regularized, it remains globally solvable and typically much better conditioned than the hard-whitened method.

4. **Closed-form Barlow Twins layer**  
   This is not an iterative optimizer at all. For `lambda > 0`, the objective is strictly convex and the solution is unique:
   `G* = lambda (M + lambda I)^(-1)`.
   So there is no training convergence issue in the usual sense; only the conditioning of `M + lambda I` matters. With `lambda = 1`, that conditioning was very benign in the current benchmark.

The main tradeoff is therefore:

- PCA is much cheaper and simpler.
- The new DNN layer is materially more expensive per layer.
- But the new layer is also much better conditioned than the old hard-whitened invariant objective and gives a unique analytic solution without any SGD loop.

For a depth-`L` greedy network, the cost of the new layer scales roughly linearly in `L`.

## References

- VICReg: <https://arxiv.org/abs/2105.04906>
- An Information-Theoretic Perspective on VICReg: <https://arxiv.org/abs/2303.00633>
- MCR2: <https://arxiv.org/abs/2006.08558>
- MEC: <https://arxiv.org/abs/2210.11464>
- Matrix Information Theory for Self-Supervised Learning: <https://proceedings.mlr.press/v235/zhang24bi.html>
- Mathematical Aspects of Deep Learning: <https://www.cambridge.org/core/books/mathematical-aspects-of-deep-learning/8D9B41D1E9BB8CA515E93412EECC2A7E>
- Locality Preserving Projections: <https://papers.nips.cc/paper/2359-locality-preserving-projections>
- Slow Feature Analysis theory notes: <https://www.ini.rub.de/PEOPLE/wiskott/Teaching/ComputationalNeuroscience/PublicWeb/SlowFeatureAnalysisTheory-S2-LectureNotes-PublicWeb.pdf>
