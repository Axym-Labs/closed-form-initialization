# Closed-Form Attention for Residual SSL/BT Networks

## Scope

This note explores how far one can push `attention` toward a closed-form or near-closed-form regime, with emphasis on our current setup:

- a **supervised residual output path**
  \[
  \hat Y_{l+1} = \hat Y_l + H_l W_l,
  \qquad
  W_l = \arg\min_W \|Y-\hat Y_l-H_lW\|_F^2 + \lambda_{\text{head}}\|W\|_F^2
  \]
- plus an **SSL hidden path** updated analytically from paired views using a Barlow/CCA-style transform.

The question is whether one can replace the current dense hidden transform by something attention-like while retaining a closed-form solve or a very small spectral optimization.


## Executive Summary

1. **Exact softmax attention with learned queries/keys/values does not admit a useful global closed-form solution in general.**
   The obstruction is not just nonconvexity; it is the combination of:
   - bilinear parameterization of `Q` and `K`
   - exponentiation / normalization by row
   - coupling between all token pairs.

2. **Closed-form attention becomes plausible once the attention weights are fixed or made analytic.**
   The main workable pattern is:
   \[
   \text{analytic attention weights} \;\to\; \text{closed-form solve for values/output map.}
   \]

3. **The most promising route for our SSL hidden path is not exact softmax attention, but a closed-form kernel/landmark attention.**
   In particular:
   - derive `Q/K` from the same shared-covariance / CCA / closed-form-BT geometry we already use
   - build a token-to-landmark attention matrix analytically
   - solve the value/output map by ridge or by the same one-parameter quadratic BT surrogate.

4. **This is closely related to several known lines of work, but the exact hybrid we want is not standard.**
   The key ingredients already exist in the literature:
   - attention as kernel smoothing
   - linear/kernel attention
   - Nyström / inducing-point attention
   - Hopfield attention / associative memory
   - probabilistic / Gaussian-process keys

5. **Implemented result in this repo: the strongest current closed-form attention block is a landmark-attention layer with spectral keys and ridge-solved values, but it still underperforms the best dense BT hidden path.**
   In the current CIFAR residual setup, attention is viable and stable, but not yet a replacement for the closed-form Barlow transform.


## Literature Scan

### 1. Original Transformer

Vaswani et al., *Attention Is All You Need*  
Source: https://arxiv.org/abs/1706.03762

For one head,
\[
\operatorname{Att}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

This already shows the main barrier to closed form:
- the attention matrix depends on `QK^T`
- then passes through row-wise softmax
- then multiplies `V`

If `Q = XW_Q`, `K = XW_K`, `V = XW_V`, the full mapping is highly coupled and nonlinear in the unknown weights.


### 2. Attention as Kernel Smoothing

Tsai et al., *Transformer Dissection: A Unified Understanding of Transformer's Attention via the Lens of Kernel*  
Source: https://arxiv.org/abs/1908.11775

This paper explicitly interprets attention as a **kernel smoother**. That is the most useful conceptual bridge for our purposes.

Takeaway for us:
- attention can be viewed as data-adaptive smoothing with kernel scores
- once viewed that way, one can replace softmax attention by other kernel constructions that are easier to analyze

This is directly relevant to a closed-form hidden path, because our BT/CCA hidden transforms are already kernel/spectral in spirit.


### 3. Linear / Kernel Attention

Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*  
Source: https://arxiv.org/abs/2006.16236

The paper rewrites attention with a positive feature map \(\phi\):
\[
\operatorname{sim}(q,k) = \phi(q)^\top \phi(k),
\]
so the output becomes
\[
\operatorname{Att}(Q,K,V)

=
\operatorname{diag}\!\big(\phi(Q)\,(\phi(K)^\top \mathbf 1)\big)^{-1}
\phi(Q)\,\phi(K)^\top V.
\]

This matters because:
- once \(\phi(Q)\) and \(\phi(K)\) are fixed, the mapping is **linear in \(V\)**
- that makes a closed-form value solve plausible

This is the strongest general template for closed-form attention in our setting.


### 4. Performer / Random Feature Approximation

Choromanski et al., *Rethinking Attention with Performers*  
Source: https://openreview.net/forum?id=Ua6zuk0WRH

Performers approximate softmax attention with positive orthogonal random features (FAVOR+). The key point for us is not the speedup itself, but the structure:

- approximate the difficult softmax kernel by a fixed/random feature map
- the remaining computation becomes linear attention

For closed-form design this suggests:
- use **analytic** or fixed feature maps, not learned ones
- solve only the value/output part


### 5. Nyström / Landmark Attention

Xiong et al., *Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*  
Source: https://arxiv.org/abs/2102.03902

Nyströmformer approximates attention using low-rank landmark structure. This is important because landmarks can, in principle, be chosen analytically:

- PCA landmarks
- CCA/shared-covariance landmarks
- class prototypes
- augmentation/prototype means

So Nyström-style attention is one natural route to a closed-form or spectral attention layer.


### 6. Inducing-Point Attention

Lee et al., *Set Transformer*  
Source: https://proceedings.mlr.press/v97/lee19d.html

Set Transformer reduces attention cost using **inducing points**, inspired by sparse Gaussian processes. In our context, the key idea is:

- attention can be mediated through a small learned or fixed set of inducing points
- if those points are chosen analytically, the whole attention block may become much closer to closed form

This is especially attractive for our residual SSL hidden path:
- use a small set of analytic prototypes instead of a full dense \(d \times d\) transform
- let depth help by changing which prototypes are emphasized


### 7. Hopfield / Associative-Memory Attention

Ramsauer et al., *Hopfield Networks is All You Need*  
Source: https://arxiv.org/abs/2008.02217

Modern Hopfield networks and attention are closely linked. The paper shows attention-like retrieval can be viewed as associative memory dynamics.

This is useful because:
- associative memory models often admit more analytic structure than fully learned attention
- prototype memories / stored patterns can be computed analytically
- the attention step becomes a retrieval operator rather than a free dense map

For our setup, a prototype-memory hidden path is a plausible closed-form alternative to the current dense BT hidden transform.


### 8. Probabilistic / Structured Keys

Nguyen et al., *Improving Transformers with Probabilistic Attention Keys*  
Source: https://proceedings.mlr.press/v162/nguyen22c.html

This paper replaces redundant heads with a mixture of Gaussian keys. The important idea is:

- the **key structure** can be regularized or parameterized through a simpler object than arbitrary learned keys

In closed-form terms, this suggests:
- choose keys from analytic prototypes or Gaussian summaries of the current hidden distribution
- then solve values/output maps closed form


### 9. Attention Kernels and Eigen-Structure

Chen et al., *Self-Attention through Kernel-Eigen Pair Sparse Variational Gaussian Processes*  
Source: https://proceedings.mlr.press/v235/chen24am.html

This paper is relevant because it explicitly uses kernel eigen/singular structure for attention. It supports the idea that:
- attention kernels have a nontrivial spectral geometry
- asymmetry can be handled through singular/eigen decomposition
- small eigenfunction sets can carry most of the useful structure

This is one of the closest existing signals that a spectral/closed-form attention construction is mathematically sensible.


### 10. Recent Theory Caveats

Ke et al., *Curse of Attention: A Kernel-Based Perspective for Why Transformers Fail to Generalize on Time Series Forecasting and Beyond*  
Source: https://proceedings.mlr.press/v280/ke25a.html

Abella et al., *Consensus Is All You Get: The Role of Attention in Transformers*  
Source: https://proceedings.mlr.press/v267/abella25a.html

These are important cautionary papers:
- attention can have **asymmetric learning / generalization failures**
- repeated attention can drive a **consensus/averaging effect**

This matters for our project because it suggests that a naive analytic attention stack may collapse to smoothing/consensus unless the residual path and prototype structure are carefully designed.


## Why Exact Softmax Attention Is Not Closed Form

Consider one layer with one head and squared loss:
\[
\mathcal L
=
\frac12 \|Y - \operatorname{softmax}(QK^\top/\sqrt d)\,V\,W_O\|_F^2.
\]

Let
\[
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V.
\]

Then
\[
A(W_Q,W_K)
:=
\operatorname{softmax}\!\left(\frac{XW_QW_K^\top X^\top}{\sqrt d}\right).
\]

So the output is
\[
\hat Y = A(W_Q,W_K)\,XW_VW_O.
\]

The difficulties are:

1. **Bilinear parameterization inside the exponent**
   \[
   XW_QW_K^\top X^\top
   \]
   is already bilinear in \(W_Q, W_K\).

2. **Row-wise softmax normalization**
   Each row is divided by a sum of exponentials depending on all tokens:
   \[
   A_{ij}
   =
   \frac{\exp(q_i^\top k_j/\sqrt d)}
        {\sum_t \exp(q_i^\top k_t/\sqrt d)}.
   \]

3. **Global coupling**
   Every token pair influences the normalization of every row.

4. **Composition with \(V W_O\)**
   Even if \(A\) were fixed, solving for \(W_V, W_O\) is only linear in a combined product, not jointly unique.

So standard softmax attention is not an eigenproblem, not a Sylvester equation, and not a simple pseudoinverse solve.


## Implemented Closed-Form Attention Variants

The current code implements several analytic attention blocks in
`closed_form_attention.py`, all designed for the same residual CIFAR setup as the retained BT/CCA hidden-path experiments.

### 1. Landmark Attention with Spectral Keys

Given paired hidden views \(H^{(1)}, H^{(2)} \in \mathbb R^{n \times d}\), form the paired second-order statistics
\[
\bar\Sigma, \qquad \mathrm{shared}.
\]
Then whiten and diagonalize the shared covariance:
\[
S = \bar\Sigma^{-1/2}\,\mathrm{shared}\,\bar\Sigma^{-1/2}
  = U \Lambda U^\top.
\]
Take the top \(L\) eigenvectors
\[
K = U_{1:L}
\]
as analytic attention keys.

For an input \(x\), define the whitened feature
\[
\tilde x = x \bar\Sigma^{-1/2},
\]
and attention weights
\[
a(x) = \operatorname{softmax}\!\left(\tau\, \tilde x K\right),
\qquad
\tau = d^{-1/2}.
\]

With values \(V \in \mathbb R^{L \times d}\), the attention output is
\[
f(x) = a(x) V.
\]

The values are solved in closed form by ridge regression. For the `mean` target mode used in the best runs,
\[
V^\star
=
\arg\min_V
\|A_1 V - T\|_F^2
\;+\;
\|A_2 V - T\|_F^2
\;+\;
\lambda \|V\|_F^2,
\]
where
\[
T = \tfrac12(H^{(1)} + H^{(2)}),
\]
and \(A_1, A_2\) are the attention-weight matrices for the two views. This has the closed form
\[
V^\star
=
\left(A^\top A + \lambda I\right)^{-1} A^\top T_{\mathrm{stack}},
\]
with the obvious stacked notation.

To preserve information, the implementation also fits a closed-form scalar blend
\[
\alpha^\star \in [0,1]
\]
so the block becomes
\[
h_{l+1} = \phi\!\big(\alpha h_l + (1-\alpha) f(h_l)\big).
\]

### 2. Axial Landmark Attention

For image-compatible widths \(d = 3 s^2\), the code also supports an axial version:
- tokenize by rows
- tokenize by columns
- fit the same closed-form landmark attention independently on each axis
- average the two outputs

This is analytically simple and more image-structured than the plain token split.

### 3. Global Landmark Attention

Instead of tokenization, one can use the full hidden vector as the query and attention over a small set of global spectral landmarks:
\[
a(x) = \operatorname{softmax}\!\left(\tau\, x \bar\Sigma^{-1/2} K\right),
\qquad
f(x) = a(x) V.
\]
This is a closed-form low-rank attention over global shared directions.

### 4. Memory Attention

The code also includes a Nyström / inducing-point style memory attention:
- choose a fixed memory bank from the shared paired features
- whiten and normalize those memories
- attend to them
- solve the values by ridge

This is closer to kernel memory regression than to spectral attention.


## Empirical Outcome of the Implemented Attention Blocks

Best current results in the recovered dual-path CIFAR setup (`width = 507`, `depth = 3`, `dual_mapping = True`):

- CIFAR-10, `random-affine`
  - `landmark-attention-mean`: `0.376`
  - `global-attention-mean`: `0.376`
  - `axial-attention-mean`: `0.366`
  - `memory-attention-mean`: `0.363`

- CIFAR-100, `random-affine`
  - `landmark-attention-mean`: `0.107`

For comparison, the current dense BT-family hidden path in the same setup reaches roughly:
- CIFAR-10, `closed-form-barlow`: `0.448`
- CIFAR-100, `closed-form-barlow`: `0.151`

So the implemented attention family is:
- mathematically coherent
- closed-form in the required sense (analytic keys + ridge-solved values)
- but not yet competitive with the best dense BT hidden path in this project

The most likely reason is that the current attention parameterization is still too low-rank / prototype-limited relative to the full dense transform, even though it is already sample-dependent.


## Implemented Closed-Form Transformer

Using the attention modules above, the repo now also contains a small **closed-form transformer-like model** for CIFAR:

- file: `transformer_cifar_compare.py`
- dataset used in the main run: CIFAR-100
- patch size: \(8 \times 8\), so \(16\) tokens per image
- token dimension: \(3 \cdot 8 \cdot 8 = 192\)
- depth: \(3\)

Each block still has the same two analytic sublayers:

1. **Closed-form token attention**
   - paired augmented token sequences \(T^{(1)}, T^{(2)}\)
   - analytic `Q/K` geometry from the paired BT statistics
   - ridge-solved output map
   - closed-form scalar blend with the identity path

2. **Closed-form feed-forward token map**
   - flatten all tokens across samples
   - fit the standard one-parameter closed-form Barlow transform on those token features
   - apply it tokenwise with a ReLU and residual addition

The supervised prediction path remains the same additive ridge residual stream:
\[
\hat Y_{l+1} = \hat Y_l + \bar T_l W_l,
\qquad
W_l = \arg\min_W \|Y - \hat Y_l - \bar T_l W\|_F^2 + \lambda_{\text{head}}\|W\|_F^2,
\]
where \(\bar T_l\) is the mean-pooled token representation before the next hidden update.

This is transformer-like in the following sense:
- tokenized input
- repeated residual attention blocks
- tokenwise feed-forward sublayer
- supervised output readout from the evolving token state

but every hidden-path parameter is still computed analytically.

The runner now supports several analytic attention families.

Domain-agnostic families:
- `landmark`
  - current baseline
  - token-to-spectral-landmark retrieval
- `spectral-self`
  - true sample-dependent token-token self-attention
  - `Q/K` come from the shared BT eigenspace rather than learned weights
- `spectral-landmark`
  - concatenate a spectral self-attention context with spectral-landmark weights
  - this tests whether prototype retrieval adds anything once token-token routing is already present

Diagnostic only:
- `local-spectral`
- `hybrid-spectral`

These locality-biased variants were useful as controls, but they are **not** the preferred direction here because the project goal is a modality-agnostic attention mechanism, not one that relies on vision-specific structure.

For the generic self-attention block, flatten token pairs across samples and compute
\[
\bar\Sigma,\qquad C,\qquad
S = \bar\Sigma^{-1/2} C \bar\Sigma^{-1/2}.
\]
Take the top shared eigenvectors \(U_r\), split them into heads \(U^{(h)}\), and define for each sample \(n\)
\[
Q_n^{(h)} = K_n^{(h)} =
\operatorname{normalize}\!\bigl(T_n \bar\Sigma^{-1/2} U^{(h)}\bigr).
\]
Then the analytic self-attention weights are
\[
A_n^{(h)} =
\operatorname{softmax}\!\left(\frac{Q_n^{(h)} K_n^{(h)\top}}{\sqrt{r_h}}\right),
\]
and the context is
\[
C_n = \operatorname{concat}_h \bigl(A_n^{(h)} T_n\bigr).
\]
Because \(A_n^{(h)}\) is analytic once \(U^{(h)}\) is fixed, the output map
\[
B^\star = \arg\min_B \|T_{\text{target}} - C B\|_F^2 + \lambda \|B\|_F^2
\]
is again just ridge regression.

This is the main conceptual upgrade over the original landmark block:
- `landmark` is prototype retrieval
- `spectral-self` is genuine token-token routing
- `spectral-landmark` is a hybrid that keeps both

The transformer runner also now supports a target scan for the analytic readout:
- `mean`
  - predict the paired token mean
- `cross`
  - predict the peer-view tokens
- `residual`
  - predict the paired residual correction
- `bt`, `bt-residual`
  - imitate the dense closed-form BT token transform, either directly or as a residual


## CIFAR-100 Comparison: Modality-Agnostic Closed-Form Transformer Variants vs Basic ViT

Full-data run:
- dataset: CIFAR-100
- suite: `random-affine`
- train/test: `10000 / 2000`
- depth: `3`
- patch size: `8`

Results:

- `landmark`, target `mean`
  - accuracy: `0.1190`
  - per-depth: `0.0590 -> 0.1045 -> 0.1190`
  - parameters: `288,000`
  - fit time: about `32.1s`

- `spectral-self`, target `mean`, `2` analytic heads (`rank = 16`)
  - accuracy: `0.1225`
  - per-depth: `0.0590 -> 0.1035 -> 0.1225`
  - parameters: `509,184`
  - fit time: about `37.2s`

- `spectral-self`, target `mean`, `1` analytic head (`rank = 8`)
  - accuracy: `0.1200`
  - per-depth: `0.0590 -> 0.1060 -> 0.1200`
  - parameters: `393,984`
  - fit time: about `30.9s`

- `spectral-landmark`, target `mean`, `2` analytic heads plus `8` landmarks
  - accuracy: `0.1195`
  - per-depth: `0.0590 -> 0.1040 -> 0.1195`
  - parameters: `518,400`
  - fit time: about `51.9s`

- learned ViT baseline (existing repo run)
  - accuracy: `0.1970`
  - per-depth: `0.1560 -> 0.1885 -> 0.1970`
  - parameters: `948,972`
  - training time: about `90.7s`
  - epochs: `20`

Reduced-data target scan (`4000 / 1000`, same depth and patch size):
- `landmark`, target `mean`: `0.095`
- `spectral-self`, target `mean`: `0.098`
- `spectral-self`, target `cross`: `0.099`
- `spectral-self`, target `bt`: `0.070`
- `spectral-landmark`, target `mean`: `0.095`
- `spectral-landmark`, target `bt-residual`: `0.070`

On the full `10000 / 2000` run, the `cross` target for `spectral-self` fell back to `0.1045`, and the residual / BT-residual targets were also worse than the plain `mean` target.

Interpretation:
- the closed-form transformer is viable and depth-helping
- **true token-token analytic self-attention helps a little**
- the gain is modest: `0.1190 -> 0.1225`, so the landmark baseline was not completely wrong, just incomplete
- the best domain-agnostic improvement is the move from prototype retrieval to sample-specific self-attention
- adding a generic prototype branch on top of self-attention does not materially help
- the dense BT transform is **not** a good teacher target for the attention block in this setup
- the remaining ViT gap is still large, so the missing ingredient is not only sample dependence; the analytic `Q/K` geometry plus one ridge-solved output map is still much less expressive than learned multihead attention

### Additional matched-budget attention scan

I also ran a second operator-level scan under the stricter rule that we should not exploit vision-specific bias or simply win by increasing model size.

Variants tested:
- `spectral-bt-context`
  - attention is used only to build token contexts
  - the attended contexts are then passed through the standard closed-form BT map
- `spectral-bt-context-centered`
  - same, but the values are centered across tokens before attention mixing
- `spectral-bt-context-weighted`
  - same, but head aggregation is weighted by the shared-spectrum strength
- `cca-self`
  - analytic asymmetric `Q/K` from the closed-form CCA pair instead of a symmetric shared eigenspace
- `spectral-self-interleaved`
  - same rank/heads as `spectral-self`, but eigendirections are interleaved across heads instead of split contiguously
- `spectral-self-whitened`
  - attention still scores in the same shared geometry, but also mixes whitened values rather than raw token values

Reduced-data outcomes (`4000 / 1000`):
- `spectral-self`: `0.098`
- `spectral-bt-context`: `0.097`
- `spectral-bt-context-centered`: `0.097`
- `spectral-bt-context-weighted`: `0.097`
- `cca-self`: `0.092`
- `spectral-self-interleaved`: `0.094`
- `spectral-self-whitened`: `0.102`

Full-data check on the only promising survivor:
- `spectral-self-whitened`, `2` analytic heads: `0.1145`
- `spectral-self-whitened`, `1` analytic head: `0.1115`

So the whitening-of-values idea helped only in the smaller-data screen and did **not** beat the plain `spectral-self` mean-target block on the full run.

Interpretation of the failed variants:
- the main benefit seems to come from **routing with raw token values**, not from pushing more of the channel transform into the attention block
- using BT after attention contexts is coherent, but in this transformer setup it did not improve on the simpler ridge readout
- asymmetric CCA-style `Q/K` did not help, so the missing expressivity is not just the lack of an asymmetric score matrix
- interleaving eigendirections across heads also did not help, so head-balancing is not the main bottleneck
- whitening the values improves conditioning on the smaller run but removes too much amplitude information on the full run

At that stage, the best fair recipe was:
- analytic spectral self-attention
- raw token values
- ridge readout to the paired mean target
- moderate head count (`2` analytic heads worked slightly better than `1`)

The later objective-tailored experiments below update that conclusion.

### Random and untrained attention baseline comparison

To separate three different effects,
- whether token-token routing itself helps
- whether the fitted closed-form readout on top of routed contexts matters
- and whether the SSL-derived spectral `Q/K` directions are actually better than generic routing directions

I added two matched-budget baselines:

- `random-self-ridge`
  - same self-attention construction and same ridge-solved output map as `spectral-self`
  - but the query/key subspace is replaced by a random orthogonal basis of the same rank
- `random-self-untrained`
  - same random routing directions
  - but no solved output map; the block just adds the random attended context back as a residual

On the smaller `4000 / 1000` scan, the ordering looked sensible:

- `spectral-self`: `0.098`
- `random-self-ridge`: `0.095`
- `random-self-untrained`: `0.092`

But on the full `10000 / 2000` CIFAR-100 run, a single random seed already matched or slightly exceeded the spectral block, so I ran a fixed-data seed sweep over the attention randomness:

- `spectral-self`, `2` analytic heads: `0.1225`
- `random-self-ridge`, `5` attention seeds: mean `0.1243`, std `0.0020`, min `0.1220`, max `0.1275`
- `random-self-untrained`, `5` attention seeds: mean `0.1027`, std `0.0002`

This changes the interpretation.

The fully untrained baseline is clearly worse, so the closed-form attention layer is doing real work. But the fact that `random-self-ridge` is at least as good as `spectral-self` means the current gain is not coming primarily from the specific SSL-derived spectral geometry. The more defensible reading is:

- sample-dependent routing is useful
- the ridge-solved output map on top of routed contexts is useful
- but the present spectral `Q/K` restriction is not yet a reliable advantage over random orthogonal routing

So the operator family still looks valid, but the present spectral choice should be treated as an unfinished hypothesis, not as the final answer.

### Objective-tailored `Q/K` experiments

The random-baseline result suggested a mismatch in the original objective.

Plain `spectral-self` chooses `Q/K` directions from the top eigenspace of the whitened shared covariance. That is the right object if we want globally invariant feature directions, but attention does not directly need globally invariant features. It needs score patterns that create useful token neighborhoods inside each sample.

That motivated two new objective families.

#### 1. Token-centered statistics

Define token fluctuations by removing the per-sample token mean:
\[
\tilde x_{it}^{(v)} = x_{it}^{(v)} - \frac1T \sum_{s=1}^T x_{is}^{(v)}.
\]

Then derive the shared eigenspace from the paired covariance of \(\tilde x\), rather than from the raw token cloud.

I tested two versions:

- `spectral-self-token-stats`
  - use token-centered paired statistics to derive the projection basis
  - but still score the **raw** tokens at attention time
- `spectral-self-token-centered`
  - use token-centered paired statistics
  - and also score the centered tokens at attention time

Reduced-data outcomes (`4000 / 1000`):

- `spectral-self-token-stats`: `0.097`
- `spectral-self-token-centered`: `0.095`

Full-data outcome (`10000 / 2000`):

- `spectral-self-token-stats`: `0.1245`

So the important distinction is that centering the **objective used to derive the basis** helps, but fully centering the actual score inputs hurts. The best version keeps raw token scoring while removing low-frequency common modes from the covariance objective.

#### 2. Score-space power objective

For a rank-1 direction \(u\), centered attention scores induce a per-sample scalar agreement term
\[
c_i(u) = \frac1T (\tilde Z_i^{(1)} u)^\top (\tilde Z_i^{(2)} u),
\]
where \(\tilde Z\) are token-centered whitened tokens.

The corresponding centered score-matrix alignment objective is
\[
\max_{\|u\|=1} \sum_i c_i(u)^2,
\]
which is no longer a second-order eigensystem, but it does admit an orthogonalized power-iteration style solve.

I used the centered spectral basis as initialization and then refined the directions by repeated updates proportional to the gradient of \(\sum_i c_i(u)^2\).

Reduced-data outcomes (`4000 / 1000`):

- `score-self-power`, `8` iterations: `0.094`
- `score-self-power`, `16` iterations: `0.094`
- `score-self-power`, `24` iterations: `0.098`
- `score-self-power`, `32` iterations: `0.096`
- `score-self-power-raw`, `24` iterations: `0.095`

Full-data outcome (`10000 / 2000`):

- `score-self-power`, `24` iterations: `0.1240`

This says two things. First, the score-space objective is meaningful: once refined enough, it improves over the original `spectral-self` full result (`0.1240` vs `0.1225`). Second, the improvement is sensitive to the iteration count and to whether score centering is preserved at application time. For this objective, the centered score representation is part of the mechanism rather than a removable implementation detail.

#### 3. Score-gain search

The score-space experiments also suggested that the fixed transformer-style score scale was arbitrary for these analytic heads. So I added a small closed-form model selection step: keep the analytic basis fixed, sweep a short grid of score gains, and pick the gain that minimizes the actual closed-form training fit after the ridge readout is solved.

This did **not** help the token-centered statistics variant on the reduced run, and it slightly hurt the mixed objective. But it mattered for the score-space power objective:

- `score-self-power-gain`, `24` power iterations, reduced-data: `0.099`
- `score-self-power-gain`, `24` power iterations, full-data: `0.1290`
- a more flexible per-head gain search on the reduced run fell back to `0.097`, so the useful improvement seems to be the global score scale rather than extra head-specific freedom

That full result is the first principled analytic attention variant that clearly beats:

- the old `spectral-self` baseline (`0.1225`)
- the untuned `score-self-power` variant (`0.1240`)
- the `spectral-self-token-stats` variant (`0.1245`)
- and the random-ridge baseline mean (`0.1243`)

So the score objective and the score scale are coupled; once the routing directions are optimized in score space, tuning the score sharpness against the same downstream closed-form objective produces a meaningful extra gain.

#### Interpretation

These experiments support the idea that attention needs a different optimality condition from the dense BT hidden map.

- the old global shared-covariance objective was too feature-centric
- deriving `Q/K` from token-centered fluctuation statistics helps
- directly optimizing view-stable centered score patterns helps more once the score scale is tuned against the downstream fit objective
- attention quality depends not only on the subspace, but also on the effective score sharpness
- the new score-space + gain-search variant is now the strongest principled attention mechanism tested here

The strongest current principled variant is therefore:

- `score-self-power-gain` at `0.1290` on the full run

with `spectral-self-token-stats` (`0.1245`) as the best purely second-order alternative.

### Why pure cross-attention does not fit cleanly here

The current transformer evaluation is a **single-stream** deployment problem:
- at train time the hidden-path fit uses paired SSL views
- at test time the model must transform one token sequence and classify it

A true cross-attention block would use something like
\[
A_{1\to2} = \operatorname{softmax}(Q_1 K_2^\top),
\qquad
A_{2\to1} = \operatorname{softmax}(Q_2 K_1^\top),
\]
which is well-defined during pair fitting but not during single-stream inference unless we introduce a second sequence at test time.

So cross-attention is mathematically natural for the SSL pair objective, but architecturally awkward for the current transformer comparison because the deployed model is not a two-stream model.


## Where Closed Form Becomes Possible

The main trick is always the same:

\[
\text{fix or analytically derive the attention weights} \;\Rightarrow\; \text{solve values/output analytically.}
\]

### Case A: Fixed Attention Matrix

Suppose some procedure gives an attention matrix \(A \in \mathbb R^{n \times m}\). Then for targets \(T\),
\[
\min_B \|T - A B\|_F^2 + \lambda \|B\|_F^2
\]
has the ridge solution
\[
B^\star = (A^\top A + \lambda I)^{-1} A^\top T.
\]

This is the simplest closed-form attention regime.


### Case B: Fixed Queries/Keys in Linear Attention

With linear attention,
\[
A(Q,K) =
\operatorname{diag}\!\big(\phi(Q)\,(\phi(K)^\top \mathbf 1)\big)^{-1}
\phi(Q)\phi(K)^\top.
\]

If \(Q, K\) are fixed or analytically derived, then again values can be solved by ridge/pseudoinverse.


### Case C: Fixed Landmarks / Inducing Points

Let \(P \in \mathbb R^{m \times d}\) be a set of landmarks/prototypes. Define
\[
A(X,P) = \operatorname{normalize}\big(\kappa(X,P)\big)
\]
for a kernel \(\kappa\). If \(P\) is fixed analytically, values can again be solved closed form.


## A Useful Taxonomy for Our Project

For our SSL hidden path, attention can be made analytic in at least four ways.

### 1. Prototype Attention from Shared-Covariance Geometry

Compute the current paired statistics from hidden views:
\[
\bar\Sigma = \frac12(\Sigma_1+\Sigma_2),
\qquad
C = \frac12(\Sigma_{12}+\Sigma_{21}).
\]

Whitened shared covariance:
\[
S = \bar\Sigma^{-1/2} C \bar\Sigma^{-1/2}.
\]

Take the top \(r\) eigenvectors \(U_r\) of \(S\). These define an analytic shared subspace.

Then define for each sample \(h_i\):
\[
q_i = U_r^\top h_i,\qquad
k_i = U_r^\top h_i.
\]

This already gives a nontrivial closed-form query/key construction derived from the same BT/CCA geometry as our current hidden path.


### 2. Landmark Attention with Analytic Prototypes

Instead of attending over all tokens, define analytic prototypes \(P\):
- class means (supervised)
- augmentation means / orbit means
- top shared-eigenspace landmarks
- fixed landmarks chosen from leverage scores or spectral partitions

Then use a kernel attention matrix
\[
A_{ij} \propto \kappa(q_i, p_j).
\]

If \(P\) is analytic, only the values need to be solved.


### 3. Closed-Form Value Solve for the Supervised Residual Path

This is the most direct fit to our residual setup.

At layer \(l\), let \(R_l = Y - \hat Y_l\) be the supervised residual. Build an analytic attention matrix \(A_l\) from the current hidden state.

Then solve
\[
B_l^\star
=
\arg\min_B \|R_l - A_l B\|_F^2 + \lambda \|B\|_F^2
=
(A_l^\top A_l + \lambda I)^{-1} A_l^\top R_l.
\]

Then update
\[
\hat Y_{l+1} = \hat Y_l + A_l B_l^\star.
\]

This is the cleanest attention analogue of the current ridge residual path.


### 4. Closed-Form SSL Value Solve for the Hidden Path

The hidden path is harder, because our BT-family objective is not just regression to a target.

But with a fixed attention operator \(A_l\), we can define new paired hidden views
\[
\tilde H^{(1)}_l = A_l H^{(1)}_l,
\qquad
\tilde H^{(2)}_l = A_l H^{(2)}_l,
\]
compute
\[
\tilde{\bar\Sigma}_l,\qquad \tilde \Delta_l,
\]
and then apply the same one-parameter closed-form BT surrogate on that attended representation:
\[
M_l = \tilde{\bar\Sigma}_l^{-1/2}\tilde\Delta_l\tilde{\bar\Sigma}_l^{-1/2},
\qquad
G_l^\star = \lambda (M_l + \lambda I)^{-1}.
\]

Then the hidden update becomes
\[
H_{l+1} = \operatorname{Norm}\!\big(\phi(\tilde H_l G_l^\star)\big).
\]

This is not exact attention training, but it is fully coherent with our current closed-form BT machinery.


## Three Concrete Closed-Form Attention Designs for Our SSL Hidden Path

### Design A: CCA/BT-Landmark Attention

1. Compute \(S = \bar\Sigma^{-1/2} C \bar\Sigma^{-1/2}\).
2. Take top \(r\) eigenvectors \(U_r\).
3. Define analytic landmarks
   \[
   P = U_r^\top H
   \]
   or a small subset/prototype set in that basis.
4. Build linear attention
   \[
   A(H) = \operatorname{normalize}\big(\phi(U_r^\top H)\phi(P)^\top\big).
   \]
5. Solve values or output map by ridge.

Pros:
- directly tied to the SSL statistics we already trust
- low-rank / prototype structure is explicit
- closed form once the landmarks are fixed


### Design B: Nyström-Closed-Form Attention

1. Build a kernel matrix in the shared-covariance basis.
2. Choose \(m\) landmarks analytically:
   - top leverage points
   - class means
   - augmentation means
   - spectral partition representatives
3. Use Nyström approximation to form \(A\).
4. Solve values by ridge.

Pros:
- preserves an attention-like token mixing
- avoids the dense \(d \times d\) hidden transform
- depth may help more naturally because landmarks can change the feature span stagewise


### Design C: Hopfield/Prototype Attention with Closed-Form Memory

Use prototypes/memory slots \(M\) as keys and values:
\[
A(H,M) = \operatorname{normalize}\big(\kappa(H,M)\big),\qquad
O(H) = A(H,M) V_M.
\]

If keys \(M\) are chosen analytically and \(V_M\) is solved by ridge/pseudoinverse, this becomes a closed-form memory-attention layer.

Pros:
- closest to associative-memory attention
- interpretable prototypes
- good fit for SSL positives / class-conditional prototypes


## What Seems Most Promising for This Project

If we want an attention analogue of the recovered BT hidden path, the strongest path is:

### Recommendation 1: Keep the residual output path exactly as it is

\[
\hat Y_{l+1} = \hat Y_l + H_l W_l,
\qquad
W_l = \arg\min_W \|Y-\hat Y_l-H_lW\|^2 + \lambda \|W\|^2.
\]

This part is already behaving like stagewise boosting and should not be replaced by attention unless necessary.

### Recommendation 2: Replace the dense hidden transform by closed-form self-attention, but keep the `Q/K` geometry open

Instead of
\[
H_{l+1} = \operatorname{Norm}(\phi(H_l T_l)),
\]
use
\[
H_{l+1}
=
\operatorname{Norm}\!\bigl(\phi(A_l(H_l)\,H_l\,B_l)\bigr),
\]
where:
- \(A_l(H_l)\) is a token-token attention matrix built from analytic BT/CCA queries and keys
- \(B_l\) is solved by ridge on the attended token contexts
- the identity path remains explicit through the residual blend

This keeps the hidden path sample-adaptive without introducing free iterative Q/K optimization.
The full transformer experiments suggest this is better than pure landmark retrieval, but the exact `Q/K` objective matters a lot. The original global shared-eigenspace construction is not the best version; token-centered and score-space objectives both improve on it.

### Recommendation 3: Match the `Q/K` objective to routing, not just to global shared variance

The earlier experiments showed the BT hidden path mattered more than PCA or reusing the supervised residual map. That suggests:
- the hidden path should still come from the SSL geometry
- but attention could be the mechanism that makes that geometry sample-adaptive

The newer attention-specific experiments make this more concrete. Plain shared-covariance eigenvectors are not the right target by themselves. Better results come from:
- deriving the basis from token-centered fluctuation statistics
- or refining it with a score-space objective that favors view-stable centered attention patterns
- and, for those score-space heads, tuning the score gain against the downstream closed-form fit rather than leaving it at the transformer default

So the cleanest next design loop is:
- output path supervised residual
- hidden path attention from a constrained query/key family whose objective is tailored to token routing
- solve the attention readout to a simple invariant target first, rather than forcing it to imitate the dense BT map
- tune the score sharpness for that objective, because analytic heads do not come with a natural default temperature
- judge new `Q/K` constructions against random orthogonal baselines, not only against landmark attention

### Recommendation 4: Keep the mechanism modality-agnostic

Locality-biased attention is useful as a diagnostic, but it should not be the main recipe here.

The preferred attention family should work for:
- images
- arbitrary vectors chunked into tokens
- sets / patch collections / tabular chunks

So the main inductive bias should come from:
- SSL-derived shared geometry
- residual identity preservation
- low-rank analytic routing

not from any assumption that neighboring image patches are special.


## A Minimal Closed-Form Attention Layer Candidate

A concrete first prototype:

1. Current hidden state \(H_l \in \mathbb R^{n \times d}\), paired views \(H_l^{(1)}, H_l^{(2)}\).
2. Compute
   \[
   S_l = \bar\Sigma_l^{-1/2} C_l \bar\Sigma_l^{-1/2}.
   \]
3. Take top \(r\) eigenvectors \(U_l\).
4. Define analytic queries/keys:
   \[
   Q_l = H_l U_l,\qquad K_l = H_l U_l.
   \]
5. Form analytic self-attention
   \[
   A_l =
   \operatorname{softmax}\!\left(\frac{Q_l K_l^\top}{\sqrt r}\right).
   \]
6. Build token contexts \(C_l = A_l H_l\) and solve the output map by ridge to a simple invariant target.

This is fully analytic once \(U_l\) is computed.


## Why This May Succeed Where Dense Closed-Form BT Layers Struggle

Our current dense BT hidden path uses one full square transform \(T_l\). That is powerful, but also brittle:
- it is global
- it is not sample-adaptive
- repeated application can over-smooth or re-filter the same modes

Analytic attention could help because:
- it is still derived from the same SSL statistics
- but it is **input-dependent** through the attention weights
- and token-token routing gives depth a more natural job than repeated global channel transforms

The recent theory caveats matter here:
- repeated attention can induce averaging/consensus
- so the attention construction should be low-rank, residual, and tied to a supervised output residual path
- the failed BT-teacher experiments also suggest attention should specialize in routing rather than trying to copy the dense BT operator directly


## What Is Probably Not Worth Pursuing

1. **Exact softmax attention with all Q/K/V learned analytically**
   - too coupled
   - no obvious spectral or pseudoinverse reduction

2. **Arbitrary fixed nonlinearities plus closed-form attention solves**
   - this recreates the same mismatch we saw with ReLU in the dense closed-form networks

3. **Attention without analytic prototype structure**
   - if keys/queries are still free dense weights, the problem is basically back to SGD territory


## Concrete Next Experiments

1. **Spectral self-attention hidden path**
   - analytic `Q/K` from the shared BT eigenspace
   - compare `1`, `2`, and `4` analytic heads at matched output-map size

2. **Target scan on the attention readout**
   - `mean`, `cross`, residual, and BT-style teacher targets
   - current evidence says `mean` is the best full-data choice, but this should be checked under other augmentation families

3. **Generic hybrid self-attention + prototype retrieval**
   - keep spectral self-attention as the main routing operator
   - test whether a lightweight prototype branch helps in non-vision domains where global memories matter more

4. **Attention + one-parameter BT hidden update**
   - use attention only to build an adaptive intermediate representation
   - keep the proven one-parameter BT surrogate as the final hidden update
   - but do not train the attention block itself to imitate the BT map directly


## Bottom Line

The literature strongly suggests that **closed-form attention is plausible only after analytic restriction of the attention geometry**:
- fixed or analytic keys/queries
- landmark/prototype structure
- kernel/linear attention rather than raw softmax optimization

For our project, the most coherent route is still:

\[
\text{supervised residual output path}
\quad+\quad
\text{closed-form token-token self-attention hidden path}.
\]

Empirically, closed-form self-attention is better than pure landmark retrieval and much better than fully untrained random routing, so the operator family is meaningful. The best current principled recipe is now a **score-space power objective plus a fitted score gain**, which reaches `0.1290` on the full CIFAR-100 transformer run. That is still far from the learned ViT baseline, but it is the first analytic attention variant here that clearly beats both the old spectral baseline and the random-ridge reference. The remaining challenge is to make that gain more robust and to understand whether the advantage comes from a genuinely better routing geometry or from a better-conditioned score distribution.
