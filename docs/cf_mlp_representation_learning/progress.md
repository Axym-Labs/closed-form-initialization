# Progress

- EXPERIMENT/ANALYSIS (2026-06-27 23:43 CEST): Added and ran
  `cf_mlp_residual_bt_route3.py` for the requested route-3 empirical pass:
  infer the residual BP-BT update law and test whether cross-layer credit
  assignment is necessary for the useful statistics. The script compares saved
  e2e residual BP-BT checkpoints against a greedy/local residual BP-BT control
  with the same block form,
  `H <- LayerNorm(H + leaky_gelu(HW))`, but trained one layer at a time with a
  BT loss on that layer's output and no cross-layer gradient. Greedy used 100
  epochs per layer, which is roughly equal layer-forward/backward compute to
  100 e2e epochs because each greedy step trains one block. Result: the strong
  "cross-layer credit is essential" story is not supported in this setup. At
  depth 24, e2e residual BP-BT reaches hidden BT/dim `0.1198`, corr-diag
  `0.721`, shared/diff `6.33`, last-layer linear accuracy `0.150`, all-layer
  PCA `0.2166`, and best layer `0.2048` at layer 5. Greedy residual BP-BT
  reaches better hidden BT/dim `0.0297`, corr-diag `0.847`, shared/diff
  `11.33`, similar last-layer accuracy `0.153`, lower all-layer PCA `0.201`,
  and better best-layer accuracy `0.2292` at layer 3. Greedy monotonically
  improves hidden BT at all 24 layers and remains nondestructive in the
  downstream readout sense, unlike non-residual BP-BT and aggressive CF
  agreement expansion. The gradient-alignment diagnostic is mostly negative:
  trained residual updates have near-zero alignment with both local hidden-BT
  gradients and final projector-BT gradients, so the useful law is not a
  literal hidden-state gradient step. Interpretation: residual structure plus
  normalization and small-ish local refinement are sufficient to produce the
  key BP-like statistics; e2e credit may still affect distribution of useful
  information across layers, but it is not necessary for effective hidden BT
  or usable final representations here. Wrote the theoretical follow-up in
  `docs/cf_mlp_representation_learning/artifacts/route3_residual_flow_note.md`.
  The proposed natural CF object is now a residual covariance-flow/Sylvester
  update on the whitened BT cross-correlation operator, not an ad hoc
  instance-geometry preservation penalty. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_route3_residual_bt_seed7/`.

- EXPERIMENT (2026-06-27 23:02 CEST): Added
  `cf_mlp_bpbt_spectral_diagnostic.py` to compare saved BP-BT checkpoints
  against CF variants in the actual spectral objects that motivate our
  layerwise rules: BT components, shared/difference trace ratio, the
  whitened paired-difference agreement spectrum, soft CF shrinkage keep-mass,
  covariance effective rank, and layer-to-layer linear novelty. This corrects
  the previous framing: the aggressive agreement-expand/fullwhiten repair is
  not merely a finite-pair overfit story; it optimizes BT by an overly
  destructive dimensional survival rule. At depth 24, residual BP-BT improves
  BT/dim from `0.5936` to `0.1198`, corr-diag from `0.248` to `0.721`, and
  shared/diff from `1.66` to `6.09`, while the low-delta cut count
  `<=0.25` only changes from `1` to `11` and soft keep-mass at lambda `0.1`
  only from `27.2` to `44.1`; effective rank changes mildly from `60.6` to
  `44.9`, with mean layer novelty only `0.0094`. By contrast,
  `plain_cf_agreement_expand_fullwhiten_relu_k192` reaches excellent train
  BT/dim `0.008263` and corr-diag `0.972`, but does so with shared/diff
  `71.1`, soft keep-mass `216.2`, effective rank collapse `507.1 -> 71.2`,
  and very high layer novelty `0.705`, matching its downstream class collapse.
  Plain CF ReLU has the opposite failure: no low-delta survival growth
  (`0 -> 0` under the same cut count) and shared/diff decays
  `2.80 -> 1.92`, so BT worsens with depth. Non-residual BP-BT is a useful
  control: depth 6/12 improve BT, but depth 24 collapses (`final BT/dim
  0.978`, effective rank `1.0`, cut count `478`), reinforcing that the
  residual BP-BT compute graph is doing a gentle refinement that our
  non-residual CF rule lacks. Design implication: next CF attempts should
  target near-identity/residual redistribution of covariance and alignment,
  probably with normalization after the residual update, not hard
  agreement-subspace selection or stronger cutting. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bpbt_spectral_diagnostic_seed7/`
  and
  `docs/cf_mlp_representation_learning/artifacts_bpbt_spectral_diagnostic_nonres_seed7/`.

- NOTE (2026-06-27 23:00 CEST): Revisit the closed-form transformer line
  after fixing the current CF-BT depth failure. The transformer failure mode
  appears qualitatively similar to the present MLP failure: performance
  worsens or fails to improve across depth, suggesting the same destructive
  layerwise invariance mechanism may be at work rather than a purely
  architecture-specific issue.

- EXPERIMENT (2026-06-27 22:37 CEST): Added
  `cf_mlp_bt_generalization.py` and extended `collect_variant_state` to retain
  held-out paired test-view activations. This revealed an important correction
  to the apparent agreement-expansion/fullwhiten BT fix. Plain ReLU has almost
  identical train/test BT behavior (`depth24 train/test BT=0.5467/0.5447`),
  so its plateau is not just train-pair overfitting. In contrast,
  `plain_cf_agreement_expand_fullwhiten_relu_k192` is a training-pair
  solution: depth24 train BT/dim is `0.008263`, but held-out test BT/dim is
  `0.9901` with corr-diag `0.006`. Expansion-only variants also overfit
  positive-pair alignment: depth24 train/test BT for k128 is
  `0.2651/0.9945`, and for k192 is `0.2639/0.9267`. The older
  agreement+shared-CCA repair similarly fails held-out pairs (`depth24
  train/test BT=0.5368/1.024`). So the current agreement-basis repairs fit
  finite training augmentation pairs rather than learning a general invariant
  map. I then ran a downstream CIFAR100 classification guardrail across
  cutting strengths k128/k192/k224/k256, with and without full whitening. This
  confirms that cutting/replacing low-agreement directions is much too
  aggressive for class-relevant representations. Plain ReLU remains better:
  final-layer accuracy is `0.1358/0.1278/0.1094` for depths `6/12/24`, and
  all-layer PCA512 is `0.1708/0.1650/0.1508`. Every expansion variant is far
  worse at the final layer, usually near chance by depth 12/24 (`~0.013` to
  `0.023`), and all-layer PCA512 is also lower. At depth 24 the best
  expansion all-layer PCA512 is expansion-only k128 at `0.1204`; the best
  fullwhiten all-layer PCA512 is only `0.1008`, while the BT-winning k192
  fullwhiten is `0.0886`. The best individual layer remains layer 1 for all
  expansion variants (`~0.135-0.143`), still below plain ReLU's layer-1
  `0.1704`. Conclusion: the "cut instead of shrink" family is useful as a
  mechanistic counterexample to flat train BT, but it is not a valid
  representation-learning fix. It overfits paired-view geometry and destroys
  downstream class signal, so future fixes need a generalizing
  nonlinearity-aware objective or regularizer rather than stronger
  agreement-basis cutting. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bt_generalization_relu_vs_expand_fullwhiten_k192_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_generalization_expansion_and_cca_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_cutting_aggressiveness_readouts_seed7/`.

- EXPERIMENT (2026-06-27 22:16 CEST): Implemented and tested the first
  nonresidual CF-BT variant that actually fixes the layerwise BT-depth
  trajectory. The math diagnosis was that pushing low-positive-agreement
  coordinates negative cannot by itself solve BT: BT standardizes every output
  coordinate, so merely shrinking a bad coordinate leaves it as a bad
  coordinate unless it is replaced. The new
  `plain_cf_agreement_expand_*_relu_k*` family therefore selects the
  highest-agreement generalized eigenspace and expands it back to width 512
  with a fixed deterministic mixing matrix before ReLU. Expansion alone fixes
  on-diagonal alignment but creates coordinate redundancy: at depth 24,
  k128/k192/k224/k256 end at BT/dim `0.2651/0.2639/0.2612/0.2533` with
  final corr-diag near `1.0` but high weighted off-diagonal terms
  (`~0.25`). Adding ordinary full whitening as the layer normalization after
  expansion removes that redundancy once the diagonal alignment has formed.
  The strongest tested point is
  `plain_cf_agreement_expand_fullwhiten_relu_k192`, whose full-data/no-TF32
  BT/dim is `0.5384/0.2456/0.008263` for depths `6/12/24`, with the final
  layer as the best layer at all three depths. The depth-24 shared/difference
  diagnostic is correspondingly strong: final BT/dim `0.008264`, corr-diag
  `0.9718`, final shared/diff `71.07`, diff fraction `0.01388`, and best
  layer `24`. This is a real fix for the "flat at first-layer level" BT
  mechanism. However, it is not yet a representation-learning win: frozen
  CIFAR100 readouts for the same k192 fullwhiten variant collapse with depth
  (`last-layer acc = 0.0520/0.0134/0.0186`, all-layer PCA512 =
  `0.1028/0.0932/0.0886`, best layer remains layer 1 at `0.1384`). Conclusion:
  the earlier plateau was caused by using all low-agreement dimensions and
  trying to shrink/cut them instead of replacing them; the remaining failure is
  that the now-working invariant objective learns a trivial/label-poor
  invariant representation. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_agreement_expand_fullwhiten_k192_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_shared_difference_agreement_expand_fullwhiten_k192_depth24_full_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_agreement_expand_fullwhiten_readouts_seed7/`.

- EXPERIMENT (2026-06-27 21:54 CEST): Removed the default `relax4x`
  schedule by running `plain_cf_relu_constinv1.0`, i.e. constant invariance
  strength across depth, on the same full-data CIFAR100/SimCLR layerwise BT
  objective diagnostic (`w=512`, depths `6/12/24`, seed `7`). This makes the
  depth trajectory worse, not better. Final BT/dim changes from the default
  relax4x ReLU baseline `0.5344/0.5581/0.5893` to
  `0.6176/0.7033/0.7301`, and the best layer becomes layer 1 for all three
  depths instead of layer 4. The component split shows the failure is mainly
  on-diagonal alignment: final corr-diag falls to `0.355/0.246/0.208`, with
  depth-24 on/off `0.6314/0.0987`. A depth-24 shared/difference diagnostic
  agrees: constant invariance reaches final shared/diff `2.339`, best
  `2.868` at layer 2, and a preactivation-to-postactivation ratio gain below
  one (`0.926`). Conclusion: relax4x is not causing the plateau; it was
  partially damping repeated non-compositional updates. Without it, the same
  local ReLU CF step keeps destroying diagonal view alignment across depth.
  Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_relu_constinv1_seed7/`
  and
  `docs/cf_mlp_representation_learning/artifacts_shared_difference_constinv_depth24_full_seed7/`.

- EXPERIMENT (2026-06-27 21:45 CEST): Added the math-led
  `cf_mlp_shared_difference_diagnostic.py`. It measures
  \(s=(x_1+x_2)/2\), \(d=(x_1-x_2)/2\), their trace ratio, diff fraction,
  and stage transition retentions across the same layer stages as the stage
  geometry diagnostic. Full-data depth-24/no-TF32 results validate the
  simplified mechanism. Plain ReLU starts with a good postnorm shared/diff
  ratio at layer 1 (`2.795`) but decays monotonically to `1.916` by layer 24,
  matching its depth plateau. Agreement bias, active-rank clipping, and
  corr-bias clipping end at only `1.483/1.354/1.434`; they preserve or add
  nonlinear novelty but do not build shared signal relative to difference
  signal. Clip-then-CCA variants also fail the ratio criterion:
  active-rank+CCA ends at `1.102`, corr-bias+CCA at `1.632`. Shared-CCA alone
  is the only current CF path that builds a large shared/diff ratio
  (`4.773` final, `9.265` best at layer 12), and this coincides with the
  good BT region, although semantic label CKA remains weak. I then tested the
  simplest principled variant implied by the note,
  `plain_cf_sharedmetric_relu`, which replaces the total-covariance shrinkage
  metric with the shared covariance metric \(\Sigma_s\). This is rejected:
  on the same full-data depth-24 diagnostic it is worse than plain ReLU
  (`shared/diff=1.774` vs `1.916`; BT/dim `0.5755` vs `0.5468`). Conclusion:
  the useful target is not another shrinkage basis by itself. A real fix must
  preserve shared coordinates through the activation/normalization step; CCA
  currently does this as a postnorm linear repair, while clipping and naive
  shared-metric shrinkage do not. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_shared_difference_depth24_full_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_shared_difference_sharedmetric_depth24_full_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts/math_first_simple_bt_relu_note.md`.

- ANALYSIS (2026-06-27 21:36 CEST): Switched from heuristic variant search
  to a math-first, stronger-simplicity loop. Wrote
  `docs/cf_mlp_representation_learning/artifacts/math_first_simple_bt_relu_note.md`.
  The simplified scalar view is \(u=s+d,\ v=s-d\): BT diagonal alignment
  requires high correlation after the activation, while off-diagonal
  decorrelation is a separate covariance problem. For a zero-mean jointly
  Gaussian scalar pair, centered ReLU correlation is a monotone function of
  the preactivation correlation; if the two scalar views are independent, any
  per-view ReLU threshold leaves their transformed coordinates independent.
  Therefore clipping low-agreement modes cannot create diagonal BT alignment;
  it can only remove variance/shared covariance. Existing full-data depth-24
  diagnostics validate this directly: active-rank and corr-bias clipping have
  much higher preactivation-to-postactivation novelty than plain ReLU
  (`0.1822/0.1892` vs `0.0083`) and lower weighted off-diagonal terms
  (`0.0136/0.0111` vs `0.0621`), but their final BT loss is almost entirely
  on-diagonal error (`0.789/0.801` on/dim, corr-diag `0.123`). Adding CCA
  after clipping nearly eliminates off-diagonal error (`0.0003` to `0.0030`)
  while still leaving corr-diag very low (`0.047` to `0.151`). Decision:
  stop adding active-rate/corr-bias schedules for now. The next simple
  validation should be an explicit shared/difference diagnostic using
  \(s=(x_1+x_2)/2\) and \(d=(x_1-x_2)/2\): a candidate layer should preserve
  \(\operatorname{tr}\Sigma_s\) while suppressing \(\operatorname{tr}\Sigma_d\)
  before relying on the nonlinearity. This points to a shared-signal-metric
  generalized-eigen shrinkage as the next principled parametrization target,
  not more clipping heuristics.

- EXPERIMENT (2026-06-27 21:29 CEST): With the GPU free, reran the missing
  three-depth layerwise BT trajectories for the new corr-bias family while
  reusing cached backprop/ReLU/active-rank controls. I added model filtering
  and a `--no-tf32` switch to `cf_mlp_bt_objective_by_layer.py` so these jobs
  compute only the new non-residual CF rows instead of recomputing the
  backprop baselines. The trajectory script still spends substantial time
  materializing layer activations and computing BT metrics through NumPy, so
  parallelism is CPU-heavy even when the CF path uses CUDA; future large
  sweeps should keep the BT metrics on torch or use the stage-geometry script.
  Results: corr-bias without CCA is a mild trajectory-shape variant but not a
  BT fix. For `b=-0.25/-0.5/-1.0`, final BT/dim over depths `6/12/24` was
  `0.835/0.8135/0.8096`, `0.8237/0.8113/0.8307`, and
  `0.8240/0.8292/0.8460`; final corr-diag stayed only around `0.10-0.12`.
  Adding shared-CCA after corr-bias confirms the composition failure across
  all three depths, not only at depth 24. With no TF32, corr-bias+CCA final
  BT/dim for `b=-0.25/-0.5/-1.0` was `0.9239/0.8793/0.9021`,
  `0.8985/0.9264/0.8546`, and `0.9580/0.8580/0.7741`; weighted off-diagonal
  error was nearly zero, but final corr-diag collapsed to `0.02-0.15`, so
  total BT is dominated by positive-pair alignment failure. The current
  no-TF32 shared-CCA control remains much better (`0.3852/0.4573/0.1980`,
  best `0.3852/0.2721/0.1745`), although depth 12 overshoots after layer 9
  via off-diagonal blow-up. Interpretation: targeted low-correlation
  clipping/biasing does not rescue the mechanism. It can preserve novelty and
  lower covariance error, but it is antagonistic to the diagonal alignment
  that shared-CCA repairs in the non-clipped agreement-bias path. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_corrbias_bneg025_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_corrbias_bneg05_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_corrbias_bneg1_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_corrbias_cca_bneg025_no_tf32_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_corrbias_cca_bneg05_no_tf32_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_corrbias_cca_bneg1_no_tf32_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_agreement_cca_no_tf32_current_seed7/`.

- EXPERIMENT (2026-06-27 21:12 CEST): Tested the natural combined fix
  suggested by the previous split: active-rank negative/clipping pressure to
  preserve new nonlinear views, followed by the shared-CCA postnorm correction
  to repair positive-pair alignment. Added
  `plain_cf_agreement_activerank_ccalinear_relu_lo*_hi*`, which reuses the
  existing active-rank bias construction and turns on
  `postnorm_linear_kind="shared_cca"`. This directly tests whether the two
  partial mechanisms compose. Result: they do not. The aggressive
  `lo=0.05, hi=0.55` setting preserves strong nonlinear novelty
  (`prelinear->postact` mean `0.1853/0.1858` under TF32/no-TF32; layer-24
  `0.127/0.129`), but CCA loses almost all positive alignment
  (`final BT/dim=0.902/0.9095`, `corr_diag=0.051/0.047`). Gentler no-TF32
  ranges also fail: `lo=0.1, hi=0.8` ends at BT/dim `0.9789` with
  `corr_diag=0.0108`, and `lo=0.25, hi=0.8` ends at `0.9509` with
  `corr_diag=0.0252`. Interpretation: the negative/clipping mechanism
  strongly preserves new-view geometry, but the resulting activation geometry
  is incompatible with the current covariance-metric CCA repair; the two
  partial fixes do not add. This narrows the solution target: we need a
  nonlinearity-aware covariance/alignment map that preserves positive-pair
  correlation while inducing persistent nonlinear novelty, not a sequential
  "clip then CCA" composition. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_cf_stage_geometry_depth24_full_extended_seed7/`.

- EXPERIMENT (2026-06-27 21:05 CEST): Reconciled the apparent
  `agreement + Adam postnorm-linear` discrepancy and extended the stage
  diagnostic to the original negative-activation/clipping intervention. The
  mismatch was not a layer-reconstruction bug: a one-layer and 24-layer
  side-by-side showed the explicit stage path matches the production
  `update_path -> normalize -> postnorm_linear` path exactly. The discrepancy
  came from numerical precision. `cf_mlp_cf_stage_geometry.py` now exposes a
  `--no-tf32` flag and writes `tf32_enabled` into stage/transition rows. Under
  the default TF32 regime used by the plotting scripts, shared-CCA at
  depth 24/full-data ends at BT/dim `0.2516` (best `0.1958`, layer 14). With
  TF32 disabled, the same run improves to final `0.1980` and best `0.1745`
  at layer 20. Adam postnorm-linear similarly improves from final `0.5337`
  under TF32 to `0.3088` without TF32, but it still does not reproduce the old
  stale artifact's `0.2002`; that older number should no longer be cited as
  current evidence. I then ran full-data depth-24 stage diagnostics for the
  proposed clipping family. Agreement-space biasing gives final-layer best
  behavior and preserves activation-stage novelty (`prelinear->postact`
  mean `0.1292`, layer-24 `0.0303`), but BT alignment remains poor
  (`final BT/dim=0.7463`, `corr_diag=0.1875`). Active-rank clipping is even
  stronger on the intended mechanism (`prelinear->postact` mean `0.1822`,
  layer-24 `0.1221`, final layer best), but also fails BT alignment
  (`final BT/dim=0.8029`, `corr_diag=0.1232`). Interpretation: pushing
  low-agreement directions into the negative/clipped region does fix the
  "new nonlinear view at depth" problem in a narrow mechanistic sense, but it
  does not by itself repair positive-pair correlation. The successful repair
  needs the covariance-metric post-nonlinearity mixing/CCA piece; clipping
  alone is a trajectory-shape/novelty fix, not a BT-objective fix. The CCA
  repair is also numerically delicate, so future claims about it must report
  TF32/precision settings. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_cf_stage_geometry_depth24_full_extended_seed7/`.
  Superseded/extended by the 21:12 active-rank + CCA composition test.

- EXPERIMENT (2026-06-27 20:50 CEST): Added
  `cf_mlp_cf_stage_geometry.py` to localize the "new views" failure inside
  each CF layer. The diagnostic records BT components and geometry novelty at
  `input`, `prelinear`, `postact`, `postnorm_before_linear`, and `postnorm`,
  plus transition CKA and bidirectional-ridge novelty for
  `input->prelinear`, `prelinear->postact`, `input->postact`,
  `postact->postnorm`, and `input->postnorm`. I initially discovered the
  transition probe was CPU-bound through NumPy ridge solves, stopped those
  runs, and moved the heavy transition CKA/R2 metrics to torch/GPU before
  relaunching the parallel full-data jobs. Full-data depth-24 runs used the
  complete 50k training split for BT metrics and 12k evenly spaced examples
  for transition geometry. Result: ordinary ReLU CF has a real nonlinear kick
  only at the first layer, but it dies with depth. Its mean base-stream
  `prelinear->postact` novelty is only `0.00828`, falling from `0.1282` at
  layer 1 to `0.00070` at layer 24; `input->postnorm` novelty similarly falls
  from `0.1586` to `0.00067`. The smoother BP-BT nonlinearity is even more
  linear (`prelinear->postact` novelty mean `0.00156`, layer-24 `0.00049`),
  so smooth activation alone is not a fix. Full whitening creates a larger
  early ReLU novelty (`0.2591` at layer 1; mean `0.0161`) but fails the BT
  objective by destroying positive alignment (`corr_diag=0.0156`,
  final BT/dim `0.9693`, best layer 1). The agreement-space + shared-CCA
  variant preserves much more nonlinear stage novelty
  (`prelinear->postact` mean `0.1262`, layer 1 `0.3492`, layer 24 `0.0234`)
  and gets the best BT trajectory among these controlled non-residual CF
  variants in this stage run (`final BT/dim=0.2516`, best `0.1958` at layer
  14, `corr_diag=0.5377`; note this stage run used `postrelu_fit_samples=2048`
  and `postrelu_steps=60`, stronger than the older `1024/40` trajectory plot
  where CCA ended at `0.3072`). Interpretation: the expectation-chain breaks
  because the ordinary shrinkage parametrization quickly enters a regime where
  the nonlinearity no longer produces new representation geometry; the later
  layers are almost linear reparameterizations and therefore cannot behave
  like progressively new BT views. Whitening-only and smooth activation attack
  the wrong part of the mechanism. The repair needs agreement-oriented
  preactivation geometry plus a post-nonlinearity covariance-metric correction,
  but this still has poor semantic alignment (`label CKA=0.0054` for the
  CCA run). The Adam postnorm-linear stage run was excluded from conclusions
  because it did not reproduce the older trajectory artifact under the same
  apparent variant; treat that as an optimizer-path fidelity issue to
  reconcile before using it as evidence. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_stage_geometry_depth12_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_stage_geometry_depth24_full_summary_seed7/`,
  and the per-variant depth-24 directories under
  `docs/cf_mlp_representation_learning/artifacts_cf_stage_geometry_depth24_full_*_seed7/`.
  Superseded/qualified by the 21:05 precision audit above for postnorm-linear
  variants.

- EXPERIMENT (2026-06-27 20:24 CEST): Re-centered the ReLU/whitening checks on
  layerwise BT trajectory and representation-geometry diagnostics rather than
  readout performance. The full ReLU BT plot already existed at
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_relu_seed7/`.
  It shows the key failure directly: residual backprop-BT improves BT
  total/dim from layer 1 to the final layer with final-layer minima at depths
  `6/12/24` (`0.6091->0.1482`, `0.6195->0.0795`,
  `0.5936->0.1198`; monotone-step fractions `1.00/0.91/0.96`), while
  ReLU CF does not follow this trajectory. Residual ReLU CF peaks at layer 4
  and then degrades (`0.5303->0.4929/0.5084/0.5339`; monotone fractions
  `0.60/0.27/0.13`), and non-residual ReLU CF also peaks at layer 4
  (`0.5483->0.5344/0.5581/0.5893`; monotone fractions
  `0.40/0.18/0.09`). The component split explains why this is the wrong
  trajectory: CF reduces the weighted off-diagonal term, but positive-pair
  on-diagonal alignment gets worse with depth. For the non-residual ReLU CF
  depth-24 run, on/dim moves `0.2741->0.5272`, corr-diag mean falls
  `0.4787->0.2852`, while weighted off/dim improves `0.2742->0.0621`.
  I also ran the full-data whitening-only baseline
  (`plain_cf_relu_fullwhiten`) at depths `6/12/24`. It is a clean negative
  control: it nearly eliminates off-diagonal covariance
  (`~1e-4` weighted off/dim), but destroys view alignment, giving final
  BT/dim `0.9610/0.9650/0.9693`, best layer 1, monotone-step fraction `0`,
  and corr-diag mean only `0.020/0.018/0.016`. Finally, a full-data depth-24
  geometry pass comparing ordinary ReLU, full-whiten, and the smoother
  BP-BT nonlinearity confirmed that the problem is not just hard ReLU
  clipping: `plain_cf_bpbt_nonlinearity` is even closer to a linear
  reparameterization (`adjacent CKA=0.9960`, bidirectional R2=`0.9996`,
  novelty=`0.0004`) than ReLU (`novelty=0.0029`). Interpretation: the
  decisive interpretability target remains a backprop-like layerwise BT
  trajectory: monotone-ish total-BT reduction driven by improving positive
  alignment, not merely whitening/decorrelation. Full-whiten and smooth
  activation alone fail that target. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_fullwhiten_seed7/`
  and
  `docs/cf_mlp_representation_learning/artifacts_layer_geometry_drift_activation_whiten_depth24_full_seed7/`.

- EXPERIMENT (2026-06-27 20:12 CEST): Added
  `cf_mlp_layer_geometry_drift.py` to test whether later layers are genuinely
  new representation states or mostly linear reparameterizations. The
  diagnostic computes adjacent-layer linear CKA, ridge predictability in both
  directions, "linear novelty" (`1 - mean bidirectional R2`), CKA to layer 1,
  CKA to raw input, and CKA to labels for base/view streams. This directly
  addresses the expectation-chain failure behind the CF plateau. Result:
  ordinary non-residual ReLU CF is indeed almost not doing depth in a
  nonlinear/geometric sense. At depth 12, base-stream adjacent CKA is `0.9896`
  and mean symmetric linear R2 is `0.9944` (`novelty=0.0056`); at depth 24 it
  becomes even more linear/reparameterization-like (`adjacent CKA=0.9947`,
  R2=`0.9969`, `novelty=0.0031`). That explains the plateau: later layers
  see inputs that are almost linearly recoverable from the previous layer, so
  the shrinkage operator is iterating within a near-fixed geometry rather than
  getting substantially new nonlinear views. Agreement-bias and CCA/Adam
  variants do create more geometry drift: depth-24 CCA/Adam base novelty is
  `0.090/0.104` and final CKA to layer 1 is only `~0.105/0.113`. This is
  comparable in scale to non-residual backprop-BT at depth 12
  (`novelty=0.149`), while residual backprop-BT changes more gradually
  (`novelty=0.027`). However, CCA/Adam CF still has weak semantic alignment:
  CKA to labels remains only `~0.020-0.022` at depth 24, well below
  backprop-BT's `~0.055-0.058` in the depth-12 comparison. Interpretation:
  the original CF plateau is a real mechanistic failure, not a normalization
  artifact; the CF shrinkage path mostly preserves a linearly predictable
  geometry. Post-ReLU covariance-metric alignment fixes the "new geometry"
  problem and BT trajectory, but not yet the "useful semantic representation"
  problem. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_layer_geometry_drift_depth12_seed7/`
  and
  `docs/cf_mlp_representation_learning/artifacts_layer_geometry_drift_cf_depth24_seed7/`.

- EXPERIMENT (2026-06-27 19:58 CEST): Tested whether the shared-CCA
  post-ReLU covariance-metric correction can be made more compositional by
  scheduling the CCA covariance power upward over depth. Added
  `plain_cf_agreement_biasopt_ccapowersched_relu_p{start}_to{end}` and
  passed the layer index into `apply_postnorm_linear_if_needed` so each layer
  records its actual `postnorm_linear_cca_power`. Depth-12 smokes rejected the
  schedules as improvements over full shared-CCA. Schedules
  `p=0.125->0.5`, `0.25->0.5`, `0.375->0.5`, and `0.425->0.5` ended at BT/dim
  `0.2265/0.2146/0.2448/0.1915`; the best one is still worse than constant
  `p=0.475` (`0.1897`) and full shared-CCA (`0.1756`). Some schedules reached
  high final corr-diag (`~0.70`), but they introduced off-diagonal cost and
  local bumps; full shared-CCA remains the best monotone closed-form point in
  this family at depth 12. Interpretation: the early-layer over-whitening
  story is not enough to beat full CCA. The useful fix seems to require the
  full covariance-metric transform at every layer, or a different closed-form
  BT-aware map rather than scalar scheduling of CCA strength. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_cca_power_schedule_smoke_seed7/`.

- EXPERIMENT (2026-06-27 19:54 CEST): Tested whether the successful
  post-nonlinearity shared-CCA correction can be weakened into a pure
  coordinate rotation or a simpler least-squares alignment. Added
  `plain_cf_agreement_biasopt_crosseig_relu`,
  `plain_cf_agreement_biasopt_ccarotate_relu`,
  `plain_cf_agreement_biasopt_ccapower_relu_p*`, and
  `plain_cf_agreement_biasopt_alignls_relu_r*`. The result is a useful
  mechanism split. Orthogonal/spectral rotation alone helps only modestly:
  depth-12 final BT/dim is `0.6796` for raw cross-eig rotation and `0.6920`
  for CCA-eigen rotation, with corr-diag only reaching about `0.19`. Thus the
  missing piece is not just feature-axis orientation. Fractional CCA power
  \(W=\Sigma^{-p}Q_p\) gives a controlled interpolation from rotation to full
  CCA: final BT/dim at depth 12 is `0.5916` for `p=0.125`, `0.5461` for
  `p=0.25`, `0.3461` for `p=0.375`, `0.2702` for `p=0.425`, `0.1897` for
  `p=0.475`, and `0.1756` for full shared-CCA (`p=0.5`). Corr-diag gain rises
  correspondingly from `+0.149` to `+0.668`. Near-full powers do not beat
  full CCA on trajectory: `p=0.475` is close but peaks at layer 11, while
  full CCA is monotone to layer 12. Closed-form LS-to-midpoint alignment with
  identity ridge was rejected as a BT surrogate: small ridge improves total
  briefly but peaks at layer 5-6; large ridge becomes the original
  agreement-bias path. Conclusion: the key fix is genuinely a
  post-ReLU covariance-metric alignment/whitening step, not just clipping,
  rotation, or pairwise midpoint alignment. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_spectral_rotate_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_cca_power_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_cca_power_nearfull_smoke_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_alignls_smoke_seed7/`.

- EXPERIMENT (2026-06-27 19:44 CEST): Tested the user's proposed
  "send low-positive-agreement directions into the negative/clipped ReLU
  region" mechanism more directly. Added two closed-form threshold variants:
  `plain_cf_agreement_activegain_relu_lo*_hi*`, which sets each agreement
  eigenmode's active target from the CF gain, and
  `plain_cf_agreement_activerank_relu_lo*_hi*`, which keeps a fixed active
  target contrast by agreement-eigenvalue rank. The gain-based version is
  rejected because gains saturate after a few layers, so targets collapse to
  nearly uniform high active rates; depth-12 final BT/dim stays around
  `0.970-0.974` and corr-diag remains near `0.025-0.027`. Rank-based
  clipping is a useful diagnostic but not a fix. Full 50k CIFAR100/SimCLR
  with `lo=0.05, hi=0.55` has the desired layer ordering:
  final/best layers `6/12/22` for depths `6/12/24` and monotone-step
  fractions `1.00/1.00/0.91`. But absolute diagonal repair is too small:
  final BT/dim is `0.8266/0.8108/0.8027`, corr-diag gain is only
  `+0.0369/+0.0499/+0.0582`, versus shared-CCA's
  `+0.321/+0.443/+0.467` and residual backprop-BT's roughly
  `+0.472/+0.585/+0.473`. Conclusion: clipping low-agreement directions is
  part of the correct trajectory-shape mechanism, but it is not sufficient.
  The missing ingredient is still post-nonlinearity coordinate alignment that
  raises paired diagonal correlation by a large amount; CCA/Adam postnorm
  alignment does this, simple ReLU thresholding does not. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_activegain_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_activerank_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_activerank_lo005_hi055_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_bt_trajectory_diagnostics_alignment_plus_activerank_seed7/`.

- EXPERIMENT (2026-06-27 19:34 CEST): Tested ridge-regularized shared-CCA
  as a narrower closed-form attempt to reduce the early-layer over-whitening
  in `plain_cf_agreement_biasopt_ccalinear_relu`. This was rejected. On a
  depth-12, 12k-sample smoke, pure shared-CCA gives monotone postnorm BT
  improvement `0.9422 -> 0.1756` with best layer 12. Adding covariance ridge
  weakens the repair rather than making it more backprop-like:
  `r=0.01` gives `0.9412 -> 0.6177` with best layer 6 and only `0.45`
  monotone-step fraction, `r=0.1` gives `0.9348 -> 0.6561`, and `r=1.0`
  gives `0.9238 -> 0.7004`. The decomposition shows why: ridge leaves
  weighted off-diagonal error tiny, but corr-diag only reaches
  `0.244/0.215/0.178` instead of pure CCA's `0.700`. This is useful
  mechanistic evidence: the CCA eigendirections are not enough; the strong
  whitening/alignment transform is doing the diagonal-correlation repair, and
  naive ridge damping removes the desired trajectory. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_cca_ridge_smoke_seed7/`.

- EXPERIMENT (2026-06-27 20:55 CEST): Replaced the Adam postnorm linear
  alignment with a closed-form shared-CCA approximation. The map centers the
  two post-activation views, whitens by their average covariance, then
  eigendecomposes the symmetric cross-covariance in that whitened space. This
  tests whether the successful postnorm linear correction can be approximated
  by a direct eigensolver rather than learned by gradient descent. Result:
  supported as a weaker but real closed-form approximation. On full 50k
  CIFAR100/SimCLR, `plain_cf_agreement_biasopt_ccalinear_relu` gives
  non-residual final BT total/dim `0.4824/0.3479/0.3072` at depths `6/12/24`,
  with best layers `6/11/22` and best depth-24 BT `0.2817`. It is weaker
  than Adam postnorm linear alignment (`0.3814/0.2877/0.2002`) because it
  over-whitens early layers (`first BT ~= 0.957`), but it preserves the
  right mechanism: on-diag error improves by `+0.508/+0.655/+0.680`,
  corr-diag mean rises by `+0.321/+0.443/+0.467`, and total-decrease
  fractions are `1.00/0.82/0.74`. Identity/CCA blends (`m=0.25/0.5/0.75`)
  did not beat pure shared-CCA in the depth-12 smoke. This moves the fix from
  "Adam subproblem" toward a CF-compatible eigensolver, but the remaining
  problem is to make the closed-form alignment less over-whitening/aggressive
  in early layers. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_shared_cca_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_cca_blend_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_agreement_ccalinear_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_bt_trajectory_diagnostics_alignment_seed7/`.

- EXPERIMENT (2026-06-27 20:20 CEST): Tested whether the remaining
  agreement-space failure is loss of information or coordinate misalignment.
  Added `cf_mlp_posthoc_bt_linear_diagnostic.py`, which freezes each layer's
  representation and fits one shared linear map applied to both views. On the
  agreement-space bias-opt path, the post-hoc map dramatically lowers BT while
  preserving the backprop-like monotone trajectory: at full 50k data,
  post-hoc final BT/dim becomes `0.3616/0.2903/0.2759` for depths `6/12/24`
  versus raw agreement-space `0.8009/0.7712/0.7463`, with positive
  post-hoc on-diag improvements `+0.351/+0.416/+0.423`. This supports the
  interpretation that agreement-space CF is not losing the invariant signal;
  it presents it in a bad coordinate geometry.

- EXPERIMENT (2026-06-27 20:35 CEST): Integrated the post-hoc finding as an
  opt-in depth-path correction,
  `plain_cf_agreement_biasopt_linearopt_relu`: each layer uses agreement-space
  ReLU thresholding, normalizes the post-activation state, then fits and
  applies one shared postnorm linear BT map before feeding the next layer.
  This is the first CF variant tested here that follows a substantially
  backprop-like layerwise BT trajectory on full 50k CIFAR100/SimCLR. Non-
  residual final BT total/dim is `0.3814/0.2877/0.2002` for depths `6/12/24`;
  best layers are `6/12/23` with depth-24 best `0.1850`. The mechanism is the
  right one: on-diag error improves by `+0.310/+0.411/+0.499`, corr-diag mean
  rises by `+0.227/+0.321/+0.412`, and monotone total-decrease fractions are
  `0.80/0.91/0.87`. Weighted offdiag increases mildly, like residual
  backprop-BT, rather than being the only source of improvement. The result is
  still not a final closed-form solution because the postnorm linear map is
  fitted by Adam; it is a strong mechanistic fix/proof-of-concept for the
  missing ingredient: post-nonlinearity coordinate alignment. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_agreement_linearopt_seed7/`.

- EXPERIMENT (2026-06-27 19:35 CEST): Recentered the CF-BT investigation on
  layerwise trajectory shape rather than final readout. Added
  `cf_mlp_bt_trajectory_analysis.py` to compute first-to-final improvement,
  monotone-step fraction, best-layer position, and on/off-diagonal component
  movement for existing BT-by-layer artifacts. The audit shows the central
  mechanistic mismatch: residual backprop-BT improves mainly by **on-diagonal
  view alignment** (`corr_diag_mean` rises strongly; on-diag error falls),
  while ordinary CF variants mostly reduce weighted off-diagonal covariance
  and let on-diagonal alignment get worse. For non-residual ReLU CF at depths
  `6/12/24`, on-diag error worsens by `-0.158/-0.205/-0.253` while weighted
  offdiag improves by `+0.172/+0.196/+0.212`; best layer remains `4` and
  depth-24 final total is worse than layer 1. This explains why CF can appear
  to make BT progress in some totals while not following the backprop-like
  learning trajectory.

- EXPERIMENT (2026-06-27 19:45 CEST): Decomposed activation-aware corrections
  by objective and basis. Added post-ReLU **bias-only full-BT optimization**,
  post-ReLU diagonal-only variants, and agreement-eigenbasis threshold variants.
  Full-scale bias-only full-BT is the best low-total CF variant so far:
  non-residual final BT total/dim improves to `0.4639/0.4647/0.4650` for
  depths `6/12/24`, beating scale+bias affine-opt
  `0.4792/0.4793/0.4803`. But it is not a depth solution: best layer remains
  `4`, monotone-step fractions are only `0.60/0.45/0.61`, and on-diagonal
  alignment still degrades (`on_improvement_abs = -0.0717/-0.0772/-0.0847`,
  `corr_diag_gain = -0.056/-0.060/-0.066`). Conversely, agreement-eigenbasis
  bias-opt has exactly the desired interpretability trajectory: axis
  concentration is `~1.0`, best layer is final for depths `6/12/24`, total BT
  is monotone, and on-diag error improves by `+0.090/+0.122/+0.148` with
  positive corr-diag gain. However it starts from and remains at poor total
  BT (`0.8009/0.7712/0.7463`) because offdiag worsens slightly and diagonal
  alignment is still weak in absolute terms. Hybrid ordinary-CF/agreement
  basis tests (`m=0.05/0.1/0.25`) did not resolve the conflict: small mixes
  lower total but still degrade diagonal and peak early; larger mixes collapse
  total. Current diagnosis: the remaining fix must combine ordinary CF's
  off-diagonal/covariance control with agreement-space's monotone diagonal
  alignment, not merely tune ReLU active rates or final readout accuracy.
  Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_bt_trajectory_diagnostics_with_biasopt_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_postrelu_biasopt_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_diag_ablation_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_agreement_threshold_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_hybrid_agreement_smoke_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_selected_trajectory_full_seed7/`.

- EXPERIMENT (2026-06-27 18:55 CEST): Tested cheap closed-form-ish
  approximations to the successful post-ReLU affine correction. Added
  opt-in variants for fixed active-rate ReLU biasing, view-correlation biasing,
  and layer-ramped active-rate biasing. Smoke diagnostics at depth `6`,
  `n_train=12000`, seed `7` rejected the simple versions: fixed active targets
  `0.8/0.9/0.95` worsened final non-residual BT/dim to
  `0.668/0.760/0.822`, and corr-bias `b=0.5/1/2` gave
  `0.536/0.582/0.660` versus ordinary ReLU `0.530` and affine-opt `0.418`.
  Inspecting the affine-opt fit showed the missing structure: the learned
  ReLU active rate is layer-dependent, clipping hard at layer 1
  (`active ~= 0.26`) and then ramping upward to `~0.88` by layer 6. A
  ramped active-rate bias (`lo=0.25`, `hi=0.8`) therefore partially fixes the
  plateau on the full 50k CIFAR100/SimCLR hidden BT objective: non-residual CF
  final total/dim improves from ReLU `0.5344/0.5581/0.5893` at depths
  `6/12/24` to `0.4940/0.4997/0.5060`, but remains behind affine-opt
  `0.4792/0.4793/0.4803` and far behind residual backprop-BT. Downstream
  clean readouts show the ramp is not a representation-depth solution:
  last-layer accuracy is stable but low (`0.1480/0.1468/0.1480`), all-layer
  PCA falls to `0.1638/0.1600/0.1600`, and best layer remains layer `1`.
  Interpretation: activation-threshold scheduling explains part of the ReLU
  plateau, but the remaining gap requires a per-coordinate/nonlocal correction
  that preserves diagonal view alignment, not only a scalar active-rate
  schedule. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_threshold_heuristics_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_affine_inspect_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_active_ramp_smoke_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_active_ramp_lo025_hi08_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_active_ramp_lo025_hi08_readouts_seed7/`.

- EXPERIMENT (2026-06-27 18:24 CEST): Tested an activation-aware local
  correction for the CF-BT depth plateau: `plain_cf_postrelu_affineopt_relu`
  keeps the usual CF matrix but fits a per-coordinate scale and bias before
  ReLU, optimizing the actual post-ReLU Barlow objective on paired views
  (`2048` fitting samples, `80` Adam steps, seed `7`). This is not a
  closed-form final answer, but a targeted upper-bound/probe for whether
  ReLU-threshold awareness is the missing mechanism. Result: supported as a
  partial fix. Non-residual CF final hidden BT total/dim improves from
  ordinary ReLU CF `0.5344/0.5581/0.5893` at depths `6/12/24` to
  `0.4792/0.4793/0.4803`, removing almost all depth drift and improving more
  at larger depth. The improvement comes mainly from lower weighted offdiag
  (`0.0712/0.0664/0.0610`) while diagonal alignment is still poor
  (`corr diag mean 0.382/0.378/0.372`), so the backprop-BT gap remains large.
  Residual CF with the same branch correction lands around
  `0.5004/0.4989/0.4989`, worse than non-residual for this metric. Clean
  readouts are also improved for the last layer: ordinary ReLU CF last-layer
  accuracy `0.1358/0.1278/0.1094` becomes
  `0.1532/0.1524/0.1520`; all-layer PCA is roughly preserved
  (`0.1688/0.1692/0.1686`). Best layer still remains layer `1`, so this is
  not a full representation-depth solution. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_postrelu_affine_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_postrelu_affine_steps80_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_postrelu_affine_s2048_steps80_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_postrelu_affine_s2048_steps80_readouts_seed7/`.

- EXPERIMENT (2026-06-27 18:02 CEST): Added
  `cf_mlp_cf_mech_debug.py` plus new CF variants in
  `cf_mlp_residual_bt_variants.py` to diagnose the non-residual CF-BT depth
  plateau. The diagnostic records each layer before the linear map, after the
  linear map, after activation, and after the layer normalization used by the
  CF path. Findings on CIFAR100/SimCLR, width/input dim `512`, seed `7`,
  depths `6/12/24`: (1) normalization is not the plateau cause for the
  hidden BT diagnostic, because BT is itself per-dimension standardized and
  post-activation vs post-normalization BT totals are identical up to
  numerical noise; (2) current full-width ReLU CF has very low alignment
  between view-difference covariance and neuron axes (`delta_axis_concentration`
  roughly `0.012-0.015` at final layers), so elementwise ReLU is not clipping
  the agreement eigenmodes directly; (3) forcing agreement-eigenmode axes
  makes that alignment `~1.0` but collapses diagonal view alignment and is much
  worse (`~0.90` final BT/dim); (4) the proposed negative agreement-gating
  ReLU intervention was rejected across gate strengths `0.25/0.5/1/2/4`.
  Best gated non-residual setting `b=4` still has final BT/dim
  `0.8076/0.7986/0.7988`, far worse than ordinary ReLU CF
  `0.5344/0.5581/0.5893`, with diagonal means only `0.125/0.133/0.133`.
  Residual agreement-gating also failed (`0.7259/0.7321/0.7409` final
  BT/dim). Schedule diagnostics rejected the simpler "later layers become too
  identity-like" fix: slower relaxation and constant invariance made final
  BT worse (`plain_cf_relu_relax2`: `0.6107/0.6707/0.6906`;
  `plain_cf_relu_constinv1.0`: `0.6176/0.7033/0.7301`). Interpretation:
  depth is failing because the local delta-shrinkage/gating objective is not
  compositional for BT. It can decorrelate/offload low-agreement directions,
  but repeated or axis-aligned suppression destroys the on-diagonal view
  alignment term. Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_debug_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_gate_beta_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_cf_mech_schedule_seed7/`,
  and
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_agreement_gate_b4_seed7/`.

- EXPERIMENT (2026-06-27 17:08 CEST): Generated the same hidden-state
  BT-objective-by-layer plots with ReLU CF variants:
  `residual_cf_branch_relu` and `plain_cf_relu`, while leaving the backprop
  curves as their actual trained architectures. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_relu_seed7/`.
  ReLU CF mostly trades diagonal view alignment for lower off-diagonal
  covariance. Residual CF-BT final total/dim is `0.4929/0.5084/0.5339` for
  depths `6/12/24`, versus leaky-GELU CF `0.5481/0.5071/0.5039`; weighted
  offdiag/dim falls strongly (`0.1196/0.0997/0.0801` vs
  `0.2736/0.2057/0.1520`), but ondiag/dim worsens (`0.3734/0.4087/0.4537`
  vs `0.2745/0.3014/0.3519`). Non-residual ReLU CF shows the same pattern
  and is only better than leaky-GELU at depth `6`; by depths `12/24` its final
  total/dim is worse (`0.5581/0.5893` vs `0.5270/0.5566`).

- IMPLEMENTATION (2026-06-27 17:02 CEST): Audited the residual BT activation
  setup. Saved residual backprop-BT checkpoints for depths `6/12/24` all use
  `activation=leaky_gelu`, `activation_alpha=0.5`, with `residual_scale=1.0`
  and layernorm enabled. The residual CF-BT branch had already used the same
  activation formula and alpha; added explicit `BP_BT_ACTIVATION` and
  `BP_BT_ACTIVATION_ALPHA` constants in `cf_mlp_residual_barlow.py`, added
  `*_bpbt_nonlinearity` aliases in `cf_mlp_residual_bt_variants.py`, and
  updated `cf_mlp_bt_objective_by_layer.py` to call the explicit aliases for
  CF-BT variants. Verification: py-compiled the three scripts and ran a small
  GPU smoke pass through the plotter at depth `6`; artifact:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_bpbt_alias_smoke/`.

- EXPERIMENT (2026-06-27 16:54 CEST): Extended
  `cf_mlp_bt_objective_by_layer.py` to include non-residual variants in the
  hidden-state BT-objective diagnostic. The four plotted curves are residual
  backprop-BT, non-residual backprop-BT, residual CF-BT, and non-residual
  CF-BT on CIFAR100/SimCLR positives with `input_dim=512`, width `512`, seed
  `7`, depths `6/12/24`, and BT lambda `0.005`. Non-residual backprop-BT
  meets the hidden-state BT objective reasonably at depths `6/12`
  (`0.1708/0.1601` final total per dim), but catastrophically fails by depth
  `24` (`0.9779` final total per dim), with almost zero off-diagonal error
  but collapsed diagonal alignment (`final diag mean 0.012`). Residual
  backprop-BT remains the best hidden-state BT solver
  (`0.1482/0.0795/0.1198`). Non-residual CF-BT is close to residual CF-BT at
  depth `12`, but worse at depths `6/24`; both CF variants remain far above
  the backprop-BT objective values. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_all_variants_seed7/`
  with PNG/PDF plots and JSONL/CSV data.

- EXPERIMENT (2026-06-27 16:44 CEST): Added and ran
  `cf_mlp_bt_objective_by_layer.py` to plot hidden-state Barlow Twins objective
  satisfaction per layer for residual backprop-BT and residual CF-BT on
  CIFAR100/SimCLR positives, `input_dim=512`, width `512`, seed `7`, depths
  `6/12/24`. The metric is computed directly on the 512D hidden paired-view
  activations without the backprop projector, using BT lambda `0.005`. Residual
  backprop-BT meets the hidden-state BT objective much better than residual
  CF-BT at every depth: final total BT loss per dim is `0.1482/0.0795/0.1198`
  for backprop-BT versus `0.5481/0.5071/0.5039` for CF-BT. Backprop-BT also
  improves the objective consistently with depth within each run, even though
  only the last layer/projector is directly optimized. CF-BT reduces weighted
  off-diagonal error with depth (`0.2736 -> 0.2057 -> 0.1520` final weighted
  offdiag/dim) but worsens the on-diagonal alignment term
  (`0.2745 -> 0.3014 -> 0.3519` final ondiag/dim), so its total objective
  remains around `0.5` per dim. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_bt_objective_by_layer_seed7/`
  with PDF/PNG plots and JSONL/CSV data.

- EXPERIMENT (2026-06-27 16:32 CEST): Added and ran
  `cf_mlp_residual_barlow.py`, a residual backprop Barlow Twins baseline with
  architecture `H <- LayerNorm(H + leaky_gelu(HW))`, on CIFAR100/SimCLR
  positives with `input_dim=512`, width `512`, seed `7`, depths `6/12/24`,
  projector dim `2048`, BT lambda `0.005`, and `100` epochs. Residual
  backprop-BT beat residual CF at every depth on the final-layer readout:
  `0.2020/0.1728/0.1500` vs residual CF `0.1700/0.1636/0.1448`. The gap was
  much larger on all-layer PCA512: residual backprop-BT
  `0.2238/0.2212/0.2164` vs residual CF `0.1868/0.1844/0.1808`. Best-layer
  probes for residual backprop-BT were `0.2164` at layer 4, `0.2116` at layer
  6, and `0.2048` at layer 5. However, residual backprop-BT still showed
  final-layer depth degradation (`0.2020 -> 0.1728 -> 0.1500`), so the
  residual architecture solves the layer-1-only pathology much better than
  plain BT/CF, but does not make the deepest final representation monotonic.
  Artifact:
  `docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7/`.

- EXPERIMENT (2026-06-27 16:20 CEST): Added and ran
  `cf_mlp_residual_bt_variants.py` on CIFAR100/SimCLR positives with
  `input_dim=512`, width `512`, seed `7`, and depths `6/12/24`. The three
  tested mechanism changes were: (1) leaky-GELU `alpha=0.5` with an OLS-fitted
  activation-inverse prior, (2) a standard nonlinear residual branch
  `H <- norm(H + leaky_gelu(H A))`, and (3) a small-step linearized BT residual
  correction. The activation-inverse prior failed badly: final-layer accuracy
  was `0.0336/0.0270/0.0254`, and the OLS inverse reconstruction degraded
  strongly with depth. The standard residual CF branch was the best result:
  final-layer accuracy `0.1700/0.1636/0.1448`, all-layer PCA512
  `0.1868/0.1844/0.1808`, and best individual layer moved to layer `6/8/8`
  rather than staying at layer 1. The linearized BT residual was stable but
  plateaued: final-layer `0.1450/0.1452/0.1464`, all-PCA512 `~0.146`, best
  layer remained layer 1, and effective rank rose substantially
  (`26.8/52.1/104.8`) without improving class usefulness. Baseline plain
  CF leaky-GELU `alpha=0.5` fell with depth (`0.1274/0.1154/0.0978` last
  layer). Interpretation: an actual residual path is the useful change; the
  linearized covariance correction preserves dimensionality but does not create
  better class representations in this setup, and the inverse-prior patch is
  rejected. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_residual_bt_variants_seed7/`.

- EXPERIMENT (2026-06-27 14:20 CEST): Clarified that "best layer" in the
  corrected clean readouts means a frozen single-layer representation readout:
  fit one supervised linear classifier on that layer's 512-dimensional hidden
  activation, with no residual stream and no PCA. For the stronger CF positive
  runs, CIFAR100/SimCLR had best layer 1 at all tested depths (`0.1722` for
  depths `6/12/24`), and Tiny ImageNet/Barlow also had best layer 1
  (`0.0658` for depths `6/12`).

- EXPERIMENT (2026-06-27 14:20 CEST): Patched `cf_mlp_barlow_clean.py` to
  support `--dataset`, `--num-classes`, multiple `--depths`, and
  `--layer-only`, so backprop Barlow Twins can be evaluated without all-layer
  PCA or representation-content diagnostics. Ran BT with the same stronger
  positive policies, one tuned configuration (`projector_dim=2048`,
  `bt_lambda=0.005`, 100 epochs), and per-layer 512D linear readouts only.
  CIFAR100/SimCLR: depth 6 best/last `0.1950/0.1280`, depth 12
  `0.1994/0.1156`, depth 24 `0.1754/0.0118`; best layer was layer 1 for all
  depths. Tiny ImageNet/Barlow: depth 6 best/last `0.0704/0.0338`, depth 12
  `0.0702/0.0304`; best layer was layer 1 for both depths. Interpretation:
  the early-layer dominance is not specific to closed-form CF; it also occurs
  for backprop-BT under the same MLP and stronger positive-pair construction.
  Artifacts:
  `docs/cf_mlp_representation_learning/artifacts_barlow_layer_only_cifar100_simclr_seed7/`
  and
  `docs/cf_mlp_representation_learning/artifacts_barlow_layer_only_tiny_barlow_seed7/`.

- IMPLEMENTATION (2026-06-27 14:10 CEST): Added explicit SSL augmentation
  policies to `cf_mlp_scalability.py` while preserving the old dataset names
  and their old mild crop/flip/translate views. New dataset aliases:
  `cifar100_simclr` uses a SimCLR-style CIFAR policy
  (`RandomResizedCrop`, horizontal flip, color jitter, grayscale, no blur);
  `cifar100_barlow` and `tinyimagenet200_barlow` use a Barlow-Twins-style
  policy with crop/flip/color jitter/grayscale plus asymmetric blur and
  solarization. Added an in-process point-data cache so repeated variants or
  depths reuse generated augmented views. `cf_mlp_clean_readouts.py` now
  accepts `--dataset` and `--num-classes`, so the clean readout experiments
  can opt into these view policies and use Tiny ImageNet metadata faithfully.

- EXPERIMENT (2026-06-27 14:10 CEST): Tested the stronger SimCLR-style
  CIFAR100 positives on the corrected clean CF readout setup, seed `7`,
  `input_dim=512`, width `512`, depths `6/12/24`, variant
  `cf:relax4:leaky0.2`. This was negative for depth refinement:
  final-layer readout was `0.1336 -> 0.1286 -> 0.1118`, all-layer PCA512 was
  `0.1748 -> 0.1668 -> 0.1528`, and the best individual layer stayed layer 1
  at `0.1722`. Compared with the old mild views, the stronger positives make
  the task harder and reduce shortcut usefulness, but they do not make the
  final layer more semantic or more useful with depth. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_augmented_views_cifar100_simclr_cf_only_seed7/`.

- EXPERIMENT (2026-06-27 14:10 CEST): Ran a bounded harder-dataset sanity
  check on local Tiny ImageNet using the Barlow-style positive-pair recipe,
  seed `7`, `n_train=50000`, `n_test=5000`, `input_dim=512`, width `512`,
  depths `6/12`, variant `cf:relax4:leaky0.2`. Absolute linear-readout
  accuracies are low, as expected for 200 classes with this flattened MLP:
  depth 6 last/all-PCA512 `0.0468/0.0668`; depth 12
  `0.0390/0.0638`; best layer stayed at `0.0658`. Interpretation: the
  harder ImageNet-like dataset plus stronger augmentations is a better
  stressor, but it still does not rescue iterative final-layer refinement
  under the current flattened MLP and current CF objective. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_augmented_views_tiny_barlow_cf_seed7/`.

- DECISION (2026-06-27 14:10 CEST): Next task/dataset direction should not
  simply be "stronger augmentations on image classification". The better next
  tests are (1) nuisance-controlled evaluation on CIFAR/Tiny, where class
  readout is measured after regressing out color, brightness, contrast, and
  coarse spatial layout; (2) a shape-biased or rendition/domain task such as
  ImageNet-R subset classification or train-on-natural/test-on-rendition,
  where color/texture/global appearance shortcuts are intentionally weakened;
  and (3) if staying on CIFAR100, use the stronger SimCLR policy and report
  both ordinary and nuisance-residualized readouts, with final-layer readout as
  the primary metric and all-layer PCA as a greedy/accumulation diagnostic.

- EXPERIMENT (2026-06-27 13:40 CEST): Added and ran
  `cf_mlp_backprop_depth_scaled.py` to compare fully trained supervised
  backprop residual MLPs against the depth-scaled CF representation setup at
  `input_dim=512`, width `512`, seed `7`, depths `6/12/24`, and 30 epochs
  with best checkpoint selected by evaluated supervised accuracy. Backprop
  supervised residual accuracy peaks at `0.2470` for depth `6`, `0.2420` for
  depth `12`, and `0.2502` for depth `24`, so full backprop gets at most a
  small supervised depth benefit in this MLP. However, its final hidden
  representations get worse with depth: last-layer frozen linear readout is
  `0.0820` at depth `6`, `0.0122` at depth `12`, and `0.0102` at depth `24`.
  All-layer PCA512 readout also declines (`0.2010 -> 0.1916 -> 0.1832`).
  The best individual hidden layer is always layer 1 (`0.2200/0.2168/0.2280`).
  Corrected residual-head ablation shows later layers have essentially no
  supervised-stream importance: for depth `24`, top corrected drops are layer
  1 `+0.0492`, layer 2 `+0.0062`, layer 4 `+0.0028`, layer 3 `+0.0014`,
  layer 6 `+0.0010`; layers 19-24 all have `+0.0000`. Interpretation:
  supervised backprop solves this architecture mostly through early residual
  heads, not by making deep final states useful. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_backprop_depth_scaled_seed7/`.

- EXPERIMENT (2026-06-27 13:40 CEST): Used the same artifact to move beyond
  CKA and ask what the large-depth CF representation is actually encoding.
  For CF `cf:relax4:leaky0.2` at depth `24`, the final layer remains strongly
  decodable for low-level image attributes: attribute-from-representation
  ridge R2 is brightness `0.747`, spatial quadrants `0.785`, RGB mean
  `0.788`, RGB std/contrast `0.632`, color-opponent `0.501`, edges `0.316`,
  and frequency bands `0.215`. Nearest-neighbor attribute ratios show the
  representation clusters global appearance most strongly: RGB mean `0.371`,
  quadrants `0.391`, brightness `0.460`, RGB std `0.623`; edge/frequency
  ratios stay weak (`~0.86`). The all-layer CF PCA512 representation preserves
  these low-level factors even more strongly (`RGB mean R2 0.997`, quadrants
  `0.995`, brightness `0.913`). Compared with depth-24 backprop, CF late
  layers are much more informative: BP layer 24 has class readout `0.0102`,
  raw R2 `-0.003`, CKA raw `0.005`, and near-zero low-level attribute R2,
  while CF layer 24 has class readout `0.1336`, raw R2 `0.608`, and CKA raw
  `0.421`. Interpretation: large-depth CF is not learning semantic class
  structure, but it is learning/retaining an augmentation-stabilized low-level
  appearance code dominated by color, brightness, coarse spatial layout, and
  contrast.

- EXPERIMENT (2026-06-27 13:19 CEST): Added and ran
  `cf_mlp_barlow_clean.py` for Barlow Twins under the same same-data-instance
  CIFAR100 positive construction used by CF. The script evaluates the same two
  clean readouts: final hidden `512` to one linear classifier, and all hidden
  layers concatenated then PCA-compressed to `512` before one linear
  classifier. On seed `7`, `input_dim=512`, width `512`, depth `6`, the saved
  BT baseline has last-layer accuracy `0.1286`, all-layer PCA512 `0.1918`,
  and best individual layer `0.2034` at layer 1. A tuned `2048`-dim projector
  with BT lambda `0.005` improved the final-layer readout to only `0.1422`;
  all-layer PCA512 was `0.1926` and the best layer was `0.2034` at layer 2.
  Lowering BT lambda to `0.001` did not help (`0.1292` last, `0.1882`
  all-layer PCA512). Interpretation: the weak BT result was not primarily a
  projector-capacity issue. BT stores useful class information in early/mixed
  layers, but the final layer is a poor linear downstream representation in
  this MLP setup. Artifact:
  `docs/cf_mlp_representation_learning/artifacts_barlow_clean_seed7/`.

- EXPERIMENT (2026-06-27 13:19 CEST): Added and ran
  `cf_mlp_cf_setup_content.py` to compare the two clean CF setups directly
  with the same content metrics used for BT. At depth `6`, seed `7`,
  `cf:relax4:leaky0.2` gives last-layer accuracy `0.1760` and all-layer
  PCA512 `0.1922`; `cf:relax4:leakygelu0.5` gives last-layer `0.1776` and
  all-layer PCA512 `0.1962`. The all-layer CF PCA representations have very
  high raw-input recoverability and CKA-to-raw (`raw R2 0.917/0.960`,
  CKA raw `0.847/0.871`) but low label CKA (`~0.037-0.040`), weak same-view
  retrieval (`0.117-0.145` top1), and near-chance class kNN purity
  (`0.042-0.045`). Compared with BT all-layer PCA, CF is more input-geometry
  preserving and lower-rank, while BT is more instance-aligned and higher-rank
  (`BT all-layer PCA view top1 0.638-0.729`, rank `120-128`, label CKA
  `~0.074`). Artifact:
  `docs/cf_mlp_representation_learning/artifacts_cf_setup_content_w512_depth6_seed7/`.

- CORRECTION (2026-06-27): Reset the core representation setup definitions.
  The current supervised representation readouts are not residual streams:
  both setups feed one 512-dimensional frozen representation into one
  supervised linear classifier. The two core setups are (1) final hidden
  activation at width `512` directly into the linear classifier, and (2) all
  hidden activations concatenated, PCA-compressed to `512`, then fed into the
  same kind of supervised linear classifier. Residual supervised-head language
  is reserved only for older diagnostic artifacts, not for the current
  PCA/representation setup.

- EXPERIMENT (2026-06-27): Implemented corrected clean readouts in
  `cf_mlp_clean_readouts.py`, using `invariance_strength` as the preferred
  convention where higher means stronger invariance. The default kept mechanism
  is `relax4`, meaning invariance strength decays by `4x` each layer
  (`old lambda_reg` would grow by `4x`). On resized CIFAR100
  (`input_dim=512`, width `512`, depth `6`, PCA/readout dim `512`, seeds
  `7/11/19`), all readouts use one supervised linear classifier only. Results:
  `cf:relax4:leaky0.2` has the best final-layer representation readout
  (`0.1763 +- 0.0031`) and nearly matches the best all-layer PCA512 readout
  (`0.1952 +- 0.0053`); `cf:relax4:leakygelu0.5` is similar on final layer
  (`0.1759 +- 0.0044`) with all-layer PCA512 `0.1945 +- 0.0036`;
  `cf:relax4:relu` gives final-layer `0.1705 +- 0.0024` and all-layer PCA512
  `0.1959 +- 0.0034`; `cf:relax4:gelu` gives final-layer
  `0.1676 +- 0.0017` and all-layer PCA512 `0.1962 +- 0.0007`; high-leak
  `leaky0.8` is a failure mode (`0.1363` final layer); whitening-only is a
  negative baseline despite high rank (`0.1046` final layer, `0.1157`
  all-layer PCA512). Artifact:
  `docs/cf_mlp_representation_learning/artifacts_clean_readouts_w512_depth6_3seed/`.

- EXPERIMENT (2026-06-27): Tested depth scaling only for smoother/gentler
  activation variants under the default `relax4` mechanism at `w=512`.
  This was a negative result for refinement. For seed `7`, `leaky0.2`
  final-layer readout fell from `0.1760` at depth `6` to `0.1534` at depth
  `12` and `0.1336` at depth `24`; all-layer PCA512 also fell from `0.1954`
  to `0.1884` and `0.1780`. `leakygelu0.5` followed the same pattern:
  final-layer `0.1776 -> 0.1534 -> 0.1232`. High-leak variants did not help:
  `leaky0.8` and `leaky0.95` stayed worse and the best layer remained layer 1.
  Artifact:
  `docs/cf_mlp_representation_learning/artifacts_clean_readouts_w512_depth_smooth_seed7/`.

- EXPERIMENT (2026-06-27): Analyzed what depth-scaled CF last layers encode
  using `cf_mlp_last_layer_content.py` at `w=512`, `relax4`, seed `7`, depths
  `6/12/24`, for `leaky0.2` and `leakygelu0.5`. The last layer does not look
  like a progressively refined class representation. For `leaky0.2`, class
  linear accuracy falls from layer 1/depth 6 `0.1818` to depth-12 last layer
  `0.1534` and depth-24 last layer `0.1336`; CKA-to-labels stays tiny and
  does not improve (`~0.033-0.041`); class kNN purity stays near chance
  (`~0.04-0.048`). Meanwhile CKA/raw-input and raw reconstruction decline
  with depth (`raw R2 0.952 -> 0.722 -> 0.608` for leaky0.2), and view
  retrieval weakens at depth 24. Rank rises, but the extra variance is not
  becoming more class-useful. Interpretation: depth is drifting away from
  early input geometry without forming a better semantic/invariant code.
  Artifact:
  `docs/cf_mlp_representation_learning/artifacts_last_layer_content_w512_depth_seed7/`.

- EXPERIMENT (2026-06-27): Tested per-layer CF invariance schedules with
  `cf_mlp_lambda_schedule.py` on resized CIFAR100 (`input_dim=512`,
  width `512`, depth `6`, `n_train=50000`, `n_test=5000`, seeds `7/11/19`).
  In the current CF transform the gain is `lambda / (eig + lambda)`, so
  smaller `lambda` means stronger invariance. Thus decaying `lambda` with
  depth increases invariance with depth, while growing `lambda` relaxes
  invariance with depth. The decay direction was not helpful: `decay_0.5`
  (`[1, .5, .25, .125, .0625, .03125]`) reduced mean final supervised
  accuracy to `0.1829 +- 0.0055` vs constant baseline `0.1886 +- 0.0020`,
  pushed last-layer effective rank down to `5.7`, and made late-half PCA
  features worse (`0.1115` vs baseline `0.1372`). This supports the earlier
  diagnosis that the default path already over-compresses rather than needing
  more invariance at depth.

- EXPERIMENT (2026-06-27): Relaxing invariance with depth helped. The best
  focused schedules were `grow_2.0` (`[1, 2, 4, 8, 16, 32]`) and `grow_4.0`
  (`[1, 4, 16, 64, 256, 1024]`). Over three seeds, `grow_2.0` reached mean
  final supervised accuracy `0.2059 +- 0.0046`, and `grow_4.0` reached
  `0.2036 +- 0.0007`, both above the constant baseline `0.1886 +- 0.0020`.
  The later representations were much less collapsed: last-layer probe
  accuracy rose from baseline `0.0916 +- 0.0055` to `0.1651 +- 0.0050`
  for `grow_2.0` and `0.1707 +- 0.0023` for `grow_4.0`; last-layer effective
  rank rose from `6.3` to about `31-32`. PCA representation quality improved
  most for late features: late-half PCA rose from `0.1372` to `0.1709`
  (`grow_2.0`) and `0.1773` (`grow_4.0`). All-layer PCA also improved modestly
  to `0.1930` and `0.1959`, respectively. Artifacts are under
  `docs/cf_mlp_representation_learning/artifacts_lambda_schedule_seed7/` and
  `docs/cf_mlp_representation_learning/artifacts_lambda_schedule_3seed/`.

- USER (2026-06-27): Corrected the interpretation of the representation
  results: the near-backprop `~19%` results at `w=512` are in the resized
  regime where `input_dim=512`, so width 512 is the full feature size. The
  earlier `~8-10%` failures are full-resolution `input_dim=3072` compressed to
  width 512. Also clarified that the full-resolution issue is not just a
  generic "bottleneck" but a constraint/incompatibility in the current
  rectangular depthwise parametrization that should eventually be lifted.

- EXPERIMENT (2026-06-27): Added
  `cf_mlp_layer_mechanistic.py` to compare CF and equal-FLOP backprop with the
  same residual-stream architecture on resized CIFAR100 (`input_dim=512`,
  width `512`, depth `6`, `n_train=50000`, `n_test=5000`, seeds `7/11/19`).
  The corrected supervised-stream interpretation is: dropping the early layer
  contribution hurts, it does not help. For CF, removing layer 1 and refitting
  a full logit correction with bias drops corrected accuracy by about
  `0.064`, layer 2 by `0.011`, while layers 3-6 are near zero or slightly
  negative. Backprop shows a similar residual-stream composition but weaker:
  layer 1 corrected drop is about `0.045`, later layers are near zero. Thus
  both systems rely heavily on the first layer in this setup; late residual
  heads mostly act as small corrections and can become harmful/noisy.

- EXPERIMENT (2026-06-27): The per-layer diagnostic suggests the late CF layers
  are losing useful representation content mechanistically, not merely
  overfitting a probe. CF per-layer ridge-probe train/test accuracies both
  decline with depth: train `0.229 -> 0.105`, test `0.183 -> 0.092`. Effective
  rank collapses from already-low `14.7` at layer 1 to `6.3` at layer 6, and
  variance concentrates: top-10 covariance directions explain `0.818` of
  layer-1 variance and `0.962` of layer-6 variance. Same-instance augmented
  views become very aligned through the middle layers: view MSE ratio vs
  shuffled views goes from `0.433` at layer 1 to `0.225` at layer 3, while
  view cosine rises from `0.531` to `0.742`. Interpretation: the current CF
  depth path is producing a very low-rank, augmentation-stable code. It is
  good at enforcing invariance, but with depth it appears to over-compress
  away class/useful variation rather than building a richer hierarchy.

- EXPERIMENT (2026-06-27): Backprop is different mechanistically. Equal-FLOP
  backprop also gets most supervised utility from layer 1, but its hidden
  states do not collapse in the same way: effective ranks are `80.6, 411.5,
  188.1, 341.1, 277.0, 304.4` across layers, and top-10 variance is much less
  concentrated after layer 1. Its later layers are not very useful in the
  residual supervised stream under the equal-FLOP budget, but they retain much
  higher-dimensional variation than CF. This separates "late layers are not
  contributing to the current supervised readout" from "late layers have
  collapsed to a tiny invariant code"; the latter is much more true for CF.

- ARTIFACT (2026-06-27): Saved four comparable model artifacts for the resized
  CIFAR100 setup (`input_dim=512`, width `512`, depth `6`, seed `7`) under
  `docs/cf_mlp_representation_learning/models_resized_seed7/`:
  `01_cf_closed_form.pt`, `02_backprop_equal_flop.pt`,
  `03_backprop_full.pt`, and `04_barlow_twins.pt`. Metrics in
  `summary.json`: CF supervised residual final accuracy `0.1906`;
  equal-FLOP supervised backprop `0.1842` at `1.04` epochs; full supervised
  backprop final `0.2306` after `30` epochs with best embedded eval checkpoint
  `0.2470` at epoch `7`; Barlow Twins trained for `100` epochs but its frozen
  final-layer ridge probe only reached `0.1286`, so this BT run is a saved
  baseline artifact, not yet a strong representation result.

- IDEA (2026-06-27): Plausible ways to lift the current rectangular projection
  incompatibility: (1) decouple invariance filtering from compression by first
  applying the full-dimensional CF operator and then PCA/stable-PCA compressing
  the resulting full code; (2) derive a rectangular "stable PCA" objective that
  chooses high-variance directions subject to low positive-pair displacement,
  rather than selecting only the most invariant whitened directions; (3)
  preserve sign information through a signed or CReLU-style rectangular map
  before ReLU destroys half of each projected coordinate; (4) use a
  Nyström/low-rank factorization of the full CF operator so width can be small
  computationally without making the hidden representation a hard
  information-discarding projection.

- EXPERIMENT (2026-06-27): Implemented and ran
  `cf_mlp_representation.py`, a runner for CF-MLP representation diagnostics.
  It uses same-data-instance positives from CIFAR100 augmentations, fits only
  the unsupervised CF depth path before representation extraction, then
  evaluates PCA-compressed depth activations with a supervised ridge linear
  probe. It also fits the supervised residual heads from the current
  classification setup for mechanistic analysis only. Artifacts are under:
  `docs/cf_mlp_representation_learning/artifacts_quick/`,
  `docs/cf_mlp_representation_learning/artifacts_resized_3seed/`,
  `docs/cf_mlp_representation_learning/artifacts_fullres_seed7/`,
  `docs/cf_mlp_representation_learning/artifacts_resized_depth3_seed7/`,
  and `docs/cf_mlp_representation_learning/artifacts_resized_depth9_seed7/`.

- EXPERIMENT (2026-06-27): Mechanistic supervised-stream analysis on resized
  CIFAR100 (`input_dim=512`, width `512`, depth `6`, `n_train=50000`,
  `n_test=5000`, seeds `7/11/19`) shows that layer 1 dominates. Mean
  cumulative supervised accuracy rises from `0.1801` at layer 1 to `0.1999`
  at layer 3, then falls to `0.1886` by layer 6. Corrected logit-space
  ablation, where one layer's contribution is removed and a full `100x100`
  linear correction plus bias is refit, gives mean corrected drops:
  layer 1 `+0.0671`, layer 2 `+0.0105`, layer 3 `+0.0012`, and layers 4-6
  slightly negative. Head matrices are not very low-rank: rank for 95% head
  Frobenius energy is about `81` for layers 1-4, `79.7` for layer 5, and
  `51.7` for layer 6. Low-rank approximating every layer head gives accuracy
  `0.1541` at rank 32, `0.1790` at rank 64, and `0.1886` at full rank 100.

- EXPERIMENT (2026-06-27): PCA-compressed representation learning is usable in
  the resized/compressed CIFAR100 setup. With all layer activations except raw
  input, per-layer path-normalized PCA features at 512 dimensions reach mean
  linear-probe accuracy `0.1894 +- 0.0048` over three seeds. The best tested
  choice is the early half / top-three important layers, reaching
  `0.1945 +- 0.0021` at 512 PCA dimensions. This beats the fixed baselines in
  the same three-seed run: raw input PCA `0.1423 +- 0.0033` and random
  path-normalized depth features `0.1797 +- 0.0060`. Normalization matters:
  raw concatenated CF activations reach only `0.1678 +- 0.0052`, while raw
  per-layer z-scored activations reach `0.1898 +- 0.0050`, comparable to the
  path-normalized features.

- EXPERIMENT (2026-06-27): Layer/depth choices show that useful information is
  early, not progressively improved by deeper CF layers. In the three-seed
  resized depth-6 run, layer-wise 512-PC probe accuracies decline from layer 1
  `0.1837` to layer 2 `0.1792`, layer 3 `0.1642`, layer 4 `0.1432`, layer 5
  `0.1175`, and layer 6 `0.0981`. Seed-7 depth checks agree: depth 3 all-layer
  512-PC features reach `0.1948`; depth 6 early-half reaches `0.1948`; depth 9
  all-layer falls to `0.1844`, while selecting the top/early three layers
  recovers `0.1948`. Current recommendation: for downstream representation
  tests, use path-normalized or per-layer-zscored activations from the first
  three layers, not all late layers.

- EXPERIMENT (2026-06-27): Full-resolution CIFAR100 at width `512` remains a
  failure mode for representation learning. At `input_dim=3072`, width `512`,
  depth `6`, `n_train=50000`, seed `7`, the best CF PCA representation was
  first-layer/path-normalized at 512 PCs with accuracy `0.1030`; all-layer
  path-normalized PCA was `0.0922`. Raw full-resolution PCA reached `0.1462`
  and a random path-normalized depth path reached `0.1728`. This confirms that
  the input-resolution/width bottleneck found in supervised classification also
  damages the unsupervised representation path.

- VERIFICATION (2026-06-27): `cf_mlp_representation.py`, `cf_mlp_scalability.py`,
  and `cf_mlp_scalability_gpu.py` compile under the CUDA-enabled Python
  environment. The representation runs completed with backend `torch-cuda`
  under the `0.40` GPU memory fraction; no representation/scalability
  experiment process was left running. Scoped `git diff --check --
  cf_mlp_representation.py docs/cf_mlp_representation_learning` passed after
  recording these results.

- USER (2026-06-27): Asked to test whether the CF-MLP setup can work for
  representation learning by PCA-compressing depth-path activations, first
  using all layer activations except raw input. Requested a mechanistic
  supervised residual-stream analysis first, including head compressibility and
  layer contribution importance via shrinking/ablating one layer and correcting
  complete prediction weight and bias. If the PCA representation is usable,
  requested exploration of layer choices and other parameters informed by the
  supervised-stream evidence. Also pointed to the read-only Primary vault note
  `01 Wissen/03 Ideen/I. Closed Form DNN.md`.
