# CF-MLP Representation Learning

## Objective

Test whether the closed-form depth path can produce a compact representation
for downstream linear probing, rather than relying on the supervised residual
classification stream.

## Requirements

- Use real image data, starting from CIFAR100.
- Use same-data-instance positives from image augmentations, not same-class
  positives.
- Analyze the supervised residual stream as grounding:
  - supervised head compressibility;
  - layer contribution importance;
  - effect of shrinking or ablating one layer's supervised contribution while
    allowing a full linear-logit correction with bias.
- Test the natural representation-learning construction:
  - collect depth-path activations from all layers except the raw input;
  - compress them with PCA;
  - evaluate with a linear probe only after representation construction.
- Check whether latent normalization matters.
- If the natural representation is usable, test layer-selection variants
  informed by the supervised-stream analysis.

## Acceptance Boundary

Produce a concise evidence report with runnable artifacts under
`docs/cf_mlp_representation_learning/artifacts/`, including enough metrics to
decide whether PCA-compressed CF depth-path features are promising for
downstream experiments.
