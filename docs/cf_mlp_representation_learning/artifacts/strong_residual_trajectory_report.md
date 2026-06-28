# Strong Residual CF-BT Trajectory Plots

Plots compare the strongest saved residual CF-BT variants against residual BP-BT controls.

Files:

- `strong_residual_bt_trajectory.png/pdf`: old-style BT trajectory plot by depth.
- `strong_residual_moment_law_d12.png/pdf`: depth-12 realized moment-law trajectory.
- `strong_residual_bt_trajectory_summary.csv`: numeric summary used by the plot.

| Setup | Depth | First train BT | Final train BT | Best train BT | Best layer | Final held-out BT | Step decrease frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BP-BT e2e residual | 6 | 0.6091 | 0.1482 | 0.1482 | 6 | n/a | 1 |
| BP-BT greedy residual | 6 | 0.5188 | 0.1857 | 0.1857 | 6 | n/a | 1 |
| CF balanced identity | 6 | 0.5713 | 0.5435 | 0.5435 | 6 | 0.5768 | 1 |
| CF old-span adaptive | 6 | 0.4874 | 0.3763 | 0.3763 | 6 | 0.391 | 1 |
| BP-BT e2e residual | 12 | 0.6195 | 0.07949 | 0.07949 | 12 | n/a | 0.9091 |
| BP-BT greedy residual | 12 | 0.5188 | 0.06948 | 0.06948 | 12 | n/a | 1 |
| CF balanced identity | 12 | 0.5713 | 0.5037 | 0.5037 | 12 | 0.5505 | 1 |
| CF gain-floor progress | 12 | 0.5624 | 0.3727 | 0.3727 | 12 | 0.3995 | 1 |
| CF identity + diag metric | 12 | 0.5712 | 0.4613 | 0.4613 | 12 | 0.4964 | 1 |
| CF old-span adaptive | 12 | 0.4874 | 0.3418 | 0.3418 | 12 | 0.3603 | 1 |
| BP-BT e2e residual | 24 | 0.5936 | 0.1198 | 0.1198 | 24 | n/a | 0.9565 |
| BP-BT greedy residual | 24 | 0.5188 | 0.02967 | 0.02967 | 24 | n/a | 1 |
| CF balanced identity | 24 | 0.5701 | 0.4527 | 0.4527 | 24 | 0.4934 | 1 |
| CF gain-floor progress | 24 | 0.5569 | 0.3394 | 0.3394 | 24 | 0.3786 | 1 |
| CF old-span adaptive | 24 | 0.5037 | 0.3173 | 0.3173 | 24 | 0.3334 | 1 |
