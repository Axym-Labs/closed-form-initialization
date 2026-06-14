# Closed-Form MLP Derivation

##### Abstract

This note derives the closed-form MLP used in this repository. The construction
is greedy and layerwise: at each depth, paired views of the current hidden
representation are summarized by second-order statistics, a linear hidden map is
fit by an analytic spectral shrinkage rule, a fixed nonlinearity is applied, and
the resulting hidden variables are re-normalized before the next layer. The
readout is also closed form: it is a ridge-regression map from hidden features to
one-hot labels, optionally added as a residual prediction head at each layer.

The central layer can be derived as a regularized invariance problem in whitened
coordinates. Given paired hidden states $h_1,h_2$, define the average covariance
$\Sigma$ and pair-difference covariance $\Delta$. Whitening by $\Sigma$ converts
the pair-difference geometry into
$$
M=\Sigma^{-1/2}\Delta\Sigma^{-1/2}.
$$
The analytic hidden map is obtained by minimizing
$$
\operatorname{tr}(G^\top M G)+\lambda\|G-I\|_F^2
$$
over a whitened linear operator $G$. This yields the exact solution
$$
G^\star=\lambda(M+\lambda I)^{-1}.
$$
Thus each generalized pair-difference mode is multiplied by the scalar gain
$$
\gamma_i=\frac{\lambda}{\mu_i+\lambda},
$$
where $\mu_i$ is its whitened disagreement eigenvalue. Stable directions
$(\mu_i\approx 0)$ are preserved, unstable directions are damped, and
$\lambda$ controls the strength of the near-identity constraint.

##### 1.  Objects in the implementation

For one hidden layer, the repository starts from three arrays:
$$
H_0,\quad H_1,\quad H_2\in\mathbb R^{n\times d}.
$$
Here $H_0$ is the base hidden representation and $H_1,H_2$ are two positive
views of the same examples. At the first layer these are built from the raw
input. At later layers they are the transformed hidden states from the previous
layer.

The main implementation path is:

- `closed_form_barlow_twins.compute_paired_stats`
- `closed_form_barlow_twins.fit_layer`
- `dual_path_residual_cifar.fit_activation_transforms`
- `broader_eval_suite.fit_closed_form_mlp_state`
- `init_finetune_realworld_eval.fit_encoder_init_mlp_state`

The paired statistics are computed after centering both views by a shared mean:
$$
\bar h
=
\frac12
\left(
\frac1n\sum_{i=1}^n h_{1,i}
+
\frac1n\sum_{i=1}^n h_{2,i}
\right),
\qquad
\tilde h_{a,i}=h_{a,i}-\bar h.
$$
Then
$$
\Sigma_1=\frac1n\tilde H_1^\top\tilde H_1,
\qquad
\Sigma_2=\frac1n\tilde H_2^\top\tilde H_2,
$$
$$
\Sigma
=
\frac12(\Sigma_1+\Sigma_2),
\qquad
C
=
\frac12
\left(
\frac1n\tilde H_1^\top\tilde H_2
+
\frac1n\tilde H_2^\top\tilde H_1
\right),
$$
and
$$
\Delta
=
\frac1n(\tilde H_1-\tilde H_2)^\top(\tilde H_1-\tilde H_2).
$$
The matrix $\Sigma$ is called `sigma_bar` in code, $C$ is called `shared`, and
$\Delta$ is called `delta`.

The identity
$$
\Delta
=
\Sigma_1+\Sigma_2
-
\frac1n\tilde H_1^\top\tilde H_2
-
\frac1n\tilde H_2^\top\tilde H_1
=
2(\Sigma-C)
$$
is useful. It says that minimizing pair-difference energy is equivalent to
maximizing shared covariance when $\Sigma$ is fixed.

##### 2.  The layer objective

The goal of the analytic hidden layer is not to solve a supervised problem
directly. It is to produce a representation that is stable across paired views
without collapsing all directions. A pure invariance objective would choose the
zero map. The repository avoids this collapse by requiring the map to remain
near the identity in whitened coordinates.

Assume for this derivation that
$$
\Sigma\succ0.
$$
In implementation, eigenvalues are floored by `REG_EPS`, so the same formulas
are applied to the regularized positive-definite matrix.

Define whitened hidden variables
$$
z_a=\Sigma^{-1/2}\tilde h_a,
\qquad a\in\{1,2\}.
$$
Their pair-difference covariance is
$$
M
=
\mathbb E[(z_1-z_2)(z_1-z_2)^\top]
=
\Sigma^{-1/2}\Delta\Sigma^{-1/2}.
$$
Since $\Delta\succeq0$, the exact population matrix $M$ is positive
semidefinite. Numerical symmetrization and eigenvalue flooring in the code are
there to preserve this ideal finite-sample geometry.

Let $G\in\mathbb R^{d\times d}$ be a linear map in whitened coordinates. A
natural regularized invariance objective is
$$
J(G)
=
\mathbb E\|G(z_1-z_2)\|_2^2
+
\lambda\|G-I\|_F^2,
\qquad
\lambda>0.
$$
The first term penalizes view disagreement after the map. The second term
prevents the degenerate solution $G=0$ by charging deviation from the identity.

Writing the first term as a trace gives
$$
\mathbb E\|G(z_1-z_2)\|_2^2
=
\operatorname{tr}
\left(
G^\top
\mathbb E[(z_1-z_2)(z_1-z_2)^\top]
G
\right)
=
\operatorname{tr}(G^\top M G).
$$
Thus
$$
J(G)
=
\operatorname{tr}(G^\top M G)
+
\lambda\operatorname{tr}((G-I)^\top(G-I)).
$$

##### 3.  Closed-form solution

Differentiate $J$ with respect to $G$. Since $M$ is symmetric,
$$
\nabla_G \operatorname{tr}(G^\top M G)=2MG,
$$
and
$$
\nabla_G \lambda\|G-I\|_F^2=2\lambda(G-I).
$$
The stationarity condition is therefore
$$
2MG+2\lambda(G-I)=0,
$$
or
$$
(M+\lambda I)G=\lambda I.
$$
Because $\lambda>0$ and $M\succeq0$, $M+\lambda I$ is strictly positive
definite, so the minimizer is unique:
$$
G^\star=\lambda(M+\lambda I)^{-1}.
$$

Equivalently, let
$$
M=Q\operatorname{diag}(\mu_1,\ldots,\mu_d)Q^\top,
\qquad
\mu_i\ge 0.
$$
Then
$$
G^\star
=
Q
\operatorname{diag}
\left(
\frac{\lambda}{\mu_1+\lambda},
\ldots,
\frac{\lambda}{\mu_d+\lambda}
\right)
Q^\top.
$$
So the layer is a spectral shrinkage map. Each whitened disagreement mode
$q_i$ receives gain
$$
\gamma_i=\frac{\lambda}{\mu_i+\lambda}.
$$

This scalar formula gives the whole interpretation:

- if $\mu_i=0$, the two views already agree in direction $q_i$, so
  $\gamma_i=1$;
- if $\mu_i\gg\lambda$, the direction mostly records view-specific nuisance
  variation, so $\gamma_i\approx0$;
- increasing $\lambda$ makes the layer closer to the identity;
- decreasing $\lambda$ makes the layer more aggressively invariant.

The code computes exactly these gains in `fit_whitened_cov_layer`, with the
minor numerical guard
$$
\gamma_i
=
\frac{\lambda}{\max(\mu_i,0)+\lambda}.
$$
For the exact objective above, the maximum is unnecessary because
$\mu_i\ge0$. It is a finite-precision safeguard.

##### 4.  Generalized-eigenvalue interpretation

The eigenvectors of
$$
M=\Sigma^{-1/2}\Delta\Sigma^{-1/2}
$$
are equivalent to generalized eigenvectors of the pair
$(\Delta,\Sigma)$. If $Mq=\mu q$ and $v=\Sigma^{-1/2}q$, then
$$
\Delta v=\mu\Sigma v.
$$
Thus $\mu$ is a normalized disagreement ratio:
$$
\mu
=
\frac{v^\top\Delta v}{v^\top\Sigma v}.
$$
The layer therefore shrinks directions according to pair-difference energy per
unit average view variance. It is not simply suppressing high raw
pair-difference directions; it is suppressing directions that are unstable
relative to their total hidden energy.

Using $\Delta=2(\Sigma-C)$, the same ratio can be written as
$$
\mu
=
2
-
2\frac{v^\top C v}{v^\top\Sigma v}.
$$
So small $\mu$ means high normalized shared covariance. In this sense, the
closed-form layer is a regularized, non-collapsing shared-view filter.

##### 5.  From the spectral map to a hidden layer

The clean variational object is the whitened map $G^\star$. The repository uses
two closely related ways to turn this into an actual hidden-layer matrix.

For a bottleneck or spectral-coordinate layer with width $k<d$, the code keeps
the $k$ columns with largest gains. Let $Q_k$ contain their eigenvectors and
let $\Gamma_k=\operatorname{diag}(\gamma_{i_1},\ldots,\gamma_{i_k})$. The row
transform is
$$
A_k=\Sigma^{-1/2}Q_k\Gamma_k,
$$
so a hidden row vector is mapped as
$$
h^+=\phi(hA_k).
$$
This is exactly the projection of the whitened representation onto the most
stable spectral coordinates, followed by the gain shrinkage and the fixed
activation $\phi$.

For the historical full-transform path, used when the requested width is at
least the current hidden dimension, the code keeps a square map
$$
A_{\mathrm{full}}
=
\Sigma^{1/2}G^\star\Sigma^{-1/2},
$$
and applies
$$
h^+=\phi(hA_{\mathrm{full}}).
$$
This square map is a similarity transport of the same whitened shrinkage
operator back into the current feature coordinate system. It preserves the
spectral gains of $G^\star$ rather than returning a reduced set of whitened
coordinates.

The bottleneck formula has the most direct variational interpretation as a
whitened spectral-coordinate representation. The full-transform formula is the
coordinate-preserving analogue used for historical comparability in this repo.
Both are governed by the same derived shrinkage gains.

##### 6.  Greedy nonlinear composition

One analytic layer is not yet an MLP. The MLP is built by repeating the same
local construction.

Let
$$
H_0^{(\ell)},H_1^{(\ell)},H_2^{(\ell)}
$$
be the base and paired-view hidden arrays at layer $\ell$. The layer fit is:
$$
A^{(\ell)}
=
\operatorname{ClosedFormLayer}
\left(
H_1^{(\ell)},H_2^{(\ell)};\lambda,k
\right).
$$
Then the three streams are advanced by the same transform:
$$
H_a^{(\ell+1)}
=
\phi\left(H_a^{(\ell)}A^{(\ell)}\right),
\qquad
a\in\{0,1,2\}.
$$
After this, the implementation optionally recenters the base stream and then
normalizes all streams using training-set statistics. In the common path, it
computes a mean and scale from the base and two view streams, then applies
$$
\widehat H_a^{(\ell+1)}
=
\frac{H_a^{(\ell+1)}-m^{(\ell+1)}}{s^{(\ell+1)}}.
$$
The next layer is fit on these normalized hidden variables.

This matters theoretically. The closed-form derivation is local to a fixed
hidden representation and a fixed pair distribution. Once a nonlinearity and
normalization are applied, the next layer sees a new pair distribution. The
deep network is therefore a greedy composition of solved local problems, not a
single global closed-form minimizer of an end-to-end neural-network objective.

##### 7.  Ridge readout

The supervised part of the closed-form MLP is a ridge-regression readout. Let
$Y\in\mathbb R^{n\times c}$ be the one-hot label matrix and let $H$ be the
current hidden representation. The ordinary readout solves
$$
\min_B
\frac12\|HB-Y\|_F^2
+
\frac{\rho}{2}\|B\|_F^2.
$$
The stationarity condition is
$$
H^\top(HB-Y)+\rho B=0,
$$
so
$$
(H^\top H+\rho I)B=H^\top Y,
$$
and hence
$$
B^\star=(H^\top H+\rho I)^{-1}H^\top Y.
$$
This is implemented as `ridge_regression`.

In the dual-mapping setting, the model accumulates residual predictions. If
$\widehat Y_\ell$ is the current cumulative prediction, the layer readout solves
$$
\min_B
\frac12
\|H_\ell B-(Y-\widehat Y_\ell)\|_F^2
+
\frac{\rho}{2}\|B\|_F^2,
$$
with solution
$$
B_\ell^\star
=
(H_\ell^\top H_\ell+\rho I)^{-1}
H_\ell^\top(Y-\widehat Y_\ell).
$$
Then
$$
\widehat Y_{\ell+1}
=
\widehat Y_\ell+H_\ell B_\ell^\star.
$$
Depending on `output_source`, $H_\ell$ is either the pre-hidden representation
or the post-hidden representation. This is why the repository can train the
hidden encoder analytically from paired views while still obtaining a supervised
classifier without backpropagation.

##### 8.  Fine-tuning initialization

For the fine-tuning experiments, the analytic matrices are copied into a PyTorch
MLP. If the NumPy row transform is $A^{(\ell)}$, the corresponding PyTorch
linear layer stores
$$
W^{(\ell)}=(A^{(\ell)})^\top,
$$
because PyTorch applies linear layers as $x(W^{(\ell)})^\top$. The ridge heads
are copied in the same transposed convention.

After copying, the model can be trained by ordinary backpropagation. Therefore
the initialization studied in the main benchmark has two distinct forms:

1. `closed-form init + CE head only`: keep the analytic encoder fixed and train
   only a linear cross-entropy head on top of its features.
2. `closed-form init + compute-matched fine-tune`: use the analytic encoder as
   initialization, then update parameters by gradient descent.

The derivation above applies to the initialization. It does not imply that the
subsequent fine-tuned model remains the solution of the same closed-form
objective.

##### 9.  What is actually guaranteed

For a single layer, fixed paired hidden distribution, fixed $\lambda>0$, and
positive-definite $\Sigma$, the following statement is exact.

Let
$$
M=\Sigma^{-1/2}\Delta\Sigma^{-1/2}.
$$
Then
$$
G^\star=\lambda(M+\lambda I)^{-1}
$$
is the unique global minimizer of
$$
J(G)
=
\operatorname{tr}(G^\top M G)+\lambda\|G-I\|_F^2.
$$
Moreover, if $M=Q\operatorname{diag}(\mu_i)Q^\top$, then $G^\star$ preserves
the eigenbasis of $M$ and applies gains
$$
\gamma_i=\frac{\lambda}{\mu_i+\lambda}.
$$

The deeper closed-form MLP inherits this exact result only layer by layer. Its
global behavior depends on:

- the quality of the paired-view construction;
- the effect of the fixed nonlinearity $\phi$;
- the normalization after each layer;
- whether the model uses the full-transform or bottleneck path;
- the ridge-head configuration;
- and whether the analytic encoder is later fine-tuned.

Thus the rigorous claim is local and spectral: each hidden layer is an exact
solution to a regularized whitened invariance problem for the current paired
hidden distribution. The full MLP is a greedy nonlinear stack of these exact
local solutions.

##### 10.  Relation to Barlow-style objectives

The name `closed-form-barlow` reflects the paired-view objective, but the actual
solved layer is not the standard gradient-trained Barlow Twins loss. The
repository's layer can be read as a second-order, analytically solvable
surrogate:
$$
\text{make paired views agree}
\qquad
\text{while staying near identity in whitened coordinates.}
$$
The pair-difference term pushes toward invariance. The identity regularizer
prevents collapse. Whitening makes the disagreement eigenvalues comparable by
measuring each direction relative to average view covariance.

This produces a simple rule:
$$
\text{keep stable normalized modes, shrink unstable normalized modes.}
$$
That rule is the mathematical core of the closed-form MLP.
