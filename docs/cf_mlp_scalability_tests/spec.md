# CF-MLP Scalability Tests

## Objective

Test whether the closed-form MLP parametrization diverges negatively from a
backprop-trained residual MLP as scale increases.

## Required Comparison

- Compare backprop residual MLP against CF-MLP on an equal-FLOP basis.
- Use the same MLP architecture for both methods.
- Use a depth stream with a supervised residual prediction at each layer.
- Do not introduce fancier architectures that scale better than the shared MLP.

## Primary Question

Does CF-MLP fall off relative to backprop at larger tested scales, or does it
remain comparable under the equal-FLOP constraint?
