# Progress

- EXPERIMENT (2026-06-27 09:15 CEST): Added a BP-only step-budget diagnostic
  for the user's question about whether backprop got enough steps. The run used
  full-resolution CIFAR100 at `n=50000`, depth 3, widths `512` and `3072`,
  three seeds, and checkpoints at `1x/2x/4x/8x` the original equal-FLOP BP step
  count. Artifacts are in
  `docs/cf_mlp_scalability_tests/artifacts_backprop_steps_diagnostic/`.

  Result: backprop got the intended equal-FLOP step budget, but not always
  enough steps for convergence. At width `512`, the original budget was
  `708` steps / `3.61` epochs and BP accuracy was `0.1963`; doubling to
  `1416` steps / `7.22` epochs improved BP to `0.2233`, while `4x` and `8x`
  overfit or plateaued around `0.202`. This means the width-512 full-res CF
  failure remains real, but the exact BP margin was understated by roughly
  `+0.027` accuracy at the best longer-BP checkpoint. At width `3072`, the
  original equal-FLOP budget was only `264` steps / `1.35` epochs and BP
  accuracy was `0.1547`; `2x/4x/8x` reached `0.1777/0.1943/0.1933`. Thus the
  earlier width-3072 "CF recovers and beats BP" result should be treated as an
  equal-FLOP observation, not as evidence that CF beats a well-trained
  backprop model at that width.

- VERIFICATION (2026-06-27 09:15 CEST): Verified the BP step diagnostic has
  `24` CUDA rows covering widths `512` and `3072` and budget multipliers
  `1/2/4/8`; no CF/backprop experiment process was left running afterward.

- EXPERIMENT (2026-06-17 12:56 CEST): Explored the user's full-resolution
  CIFAR100 signal after the resized-CIFAR sweep. The harder real-image GPU
  sweep used full-resolution CIFAR100 (`input_dim=3072`) and Tiny ImageNet
  under the same equal-FLOP policy and the same depth-stream residual MLP
  architecture. Artifacts:
  `docs/cf_mlp_scalability_tests/artifacts_harder/`,
  `docs/cf_mlp_scalability_tests/artifacts_fullres_diagnostic/`, and
  `docs/cf_mlp_scalability_tests/artifacts_fullres_width50k/`.

  Result: full-resolution CIFAR100 at width `512` is indeed much worse for
  CF-MLP than backprop. At `n=6000`, full-res CIFAR100 had CF accuracy
  `0.0377` vs backprop `0.1287` (`-0.0910` gap, CF/BP error ratio `1.104`).
  At `n=50000`, the gap worsened to CF `0.0810` vs backprop `0.1963`
  (`-0.1153`, error ratio `1.144`). This is not a depth-only failure: at
  full-res `n=6000`, depth 3/9/18 were all around `-0.09`.

  The diagnostic shows the failure is strongly tied to the input-resolution to
  width bottleneck. Holding width `512`, depth 3, and the same FLOP policy,
  increasing CIFAR100 input features from `512` to `3072` flipped the `n=50000`
  gap from `+0.0087` to `-0.1153`; at `n=6000`, it flipped from `+0.0220` to
  `-0.0910`. Ridge lambda tuning did not rescue the full-res width-512 model:
  the best tested lambda was `0.1`, with CF `0.0463` vs backprop `0.1287`
  (`-0.0823`).

  A targeted width stress check at full-resolution `n=50000` showed the
  deficit mostly disappears when width reaches the input dimension. Width 256,
  512, 1024, and 2048 were still negative (`-0.2103`, `-0.1153`, `-0.0943`,
  `-0.0593`), but width `3072` reached CF `0.1900` vs backprop `0.1547`
  (`+0.0353`, error ratio `0.958`). The current interpretation is therefore
  not "CF falls off with data scale in general"; it is "CF falls off badly
  when full-resolution inputs are compressed through a narrow closed-form
  width-512 stream, and the failure is largely removed by matching hidden
  width to input dimension."

- EXPERIMENT (2026-06-17 12:56 CEST): Tiny ImageNet was harder than resized
  CIFAR100 but less diagnostic than full-res CIFAR100. At input width `1024`,
  depth 3, and data scale `n=10000/50000/100000`, CF gaps were `-0.0144`,
  `-0.0139`, and `-0.0184`. The depth axis showed a larger collapse only at
  depth 18 (`-0.0488`). This suggests CIFAR100-resized-to-512 was too simple
  or too compressed, but Tiny ImageNet did not expose as clean a failure mode
  as full-resolution CIFAR100.

- VERIFICATION (2026-06-17 12:56 CEST): Verified the generated artifact
  invariants with fresh reads: `artifacts/` contains `96` raw rows and `48`
  paired rows; `artifacts_harder/` contains `72` raw rows and `36` paired
  rows; `artifacts_fullres_diagnostic/` contains `108` raw rows and `54`
  paired rows; `artifacts_fullres_width50k/` contains `30` raw rows and `15`
  paired rows. All rows have backend `torch-cuda`. `py_compile` passed for
  `cf_mlp_scalability.py` and `cf_mlp_scalability_gpu.py`. Scoped
  `git diff --check -- cf_mlp_scalability.py cf_mlp_scalability_gpu.py
  docs/cf_mlp_scalability_tests` passed.

- USER (2026-06-17): Noted that "full-resolution CIFAR100 is much worse for
  CF than backprop" is a useful signal and asked to explore it further after
  the current exploration.

- EXPERIMENT (2026-06-17): Corrected the scalability sweep to use CIFAR100
  only, after the user rejected synthetic-data evidence. The current
  `cf_mlp_scalability.py` / `cf_mlp_scalability_gpu.py` load the shared local
  CIFAR100 cache, convert images to the historical resized flattened MLP
  feature width (`512`), build NumPy random crop/flip/translation paired views,
  and compare analytic CF-MLP against a backprop residual MLP with the same
  depth stream and cumulative post-layer residual heads. Backprop is trained
  for the largest integer number of Adam minibatch steps not exceeding the CF
  FLOP proxy. The final aggressive run used CUDA PyTorch on the RTX 5090 with
  a `0.40` per-process memory fraction, sequential execution, and thread caps
  so as not to destabilize other GPU work. Full artifacts are in
  `docs/cf_mlp_scalability_tests/artifacts/`.

  Main corrected result: on CIFAR100, CF-MLP did not fall off on the aggressive
  data-scaling axis through the full `50k` train split. It moved from a small
  deficit at `n=1000` (`-0.0037` accuracy gap, CF/BP error ratio `1.004`) to
  a small advantage at `n=50000` (`+0.0163`, error ratio `0.980`), with the
  best mean gap around `n=25000` (`+0.0237`). It did show a mild depth falloff:
  depth 2 was `+0.0123`, depth 3 was `+0.0180`, depth 6 was `+0.0067`, depth
  9 was parity, depth 12 was `-0.0117`, and depth 18 was `-0.0083`. Width did
  not show a monotone negative divergence: bottleneck widths 128/256/384 were
  negative (`-0.0360`, `-0.0370`, `-0.0410`), while historical full width 512
  was positive (`+0.0180`).

- VERIFICATION (2026-06-17): Ran the full aggressive GPU CIFAR100-only sweep
  and verified that the artifact set contains `96` raw model rows, `48` paired
  comparison rows, and `3` trend rows, all with dataset `cifar100` and backend
  `torch-cuda`; key trend assertions pass. `py_compile` passed for both runner
  scripts. Scoped `git diff --check -- cf_mlp_scalability.py
  cf_mlp_scalability_gpu.py docs/cf_mlp_scalability_tests` passed.

- USER (2026-06-17): Requested scalability tests for CF-MLP vs backprop
  residual MLP on an equal-FLOP basis, using the same depth-stream architecture
  with supervised residuals at each layer, and asked for the result: whether
  CF-MLP fell off or not.
