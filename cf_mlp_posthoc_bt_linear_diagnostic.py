import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from cf_mlp_cf_mech_debug import corr_metrics
from cf_mlp_residual_bt_variants import barlow_loss_torch, collect_variant_state
from cf_mlp_scalability import SweepPoint, write_jsonl


def fit_shared_linear_bt(x1_np, x2_np, args, device):
    n = x1_np.shape[0]
    sample_count = min(int(args.fit_samples), n)
    if sample_count < n:
        idx_np = np.linspace(0, n - 1, sample_count).astype(np.int64)
        fit1_np = x1_np[idx_np]
        fit2_np = x2_np[idx_np]
    else:
        fit1_np = x1_np
        fit2_np = x2_np
    fit1 = torch.from_numpy(np.asarray(fit1_np, dtype=np.float32)).to(device)
    fit2 = torch.from_numpy(np.asarray(fit2_np, dtype=np.float32)).to(device)
    full1 = torch.from_numpy(np.asarray(x1_np, dtype=np.float32)).to(device)
    full2 = torch.from_numpy(np.asarray(x2_np, dtype=np.float32)).to(device)
    dim = fit1.shape[1]
    eye = torch.eye(dim, dtype=fit1.dtype, device=device)
    delta = torch.nn.Parameter(torch.zeros((dim, dim), dtype=fit1.dtype, device=device))
    optimizer = torch.optim.Adam([delta], lr=float(args.lr))
    last_loss = float("nan")
    for _ in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        w = eye + delta
        z1 = fit1 @ w
        z2 = fit2 @ w
        bt, _, _ = barlow_loss_torch(z1, z2, args.bt_lambda)
        ridge = torch.mean(delta * delta)
        loss = bt + float(args.ridge) * ridge
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([delta], float(args.grad_clip))
        optimizer.step()
        last_loss = float(loss.detach().cpu().item())
    with torch.no_grad():
        w = eye + delta
        z1 = full1 @ w
        z2 = full2 @ w
        metrics = corr_metrics(z1, z2, args.bt_lambda)
        fro_delta = torch.linalg.matrix_norm(delta).detach().cpu().item()
        fro_w = torch.linalg.matrix_norm(w).detach().cpu().item()
    metrics = {f"posthoc_{key}": value for key, value in metrics.items()}
    metrics.update(
        {
            "posthoc_fit_loss": float(last_loss),
            "posthoc_fit_samples": int(sample_count),
            "posthoc_steps": int(args.steps),
            "posthoc_lr": float(args.lr),
            "posthoc_ridge": float(args.ridge),
            "posthoc_fro_delta": float(fro_delta),
            "posthoc_fro_w": float(fro_w),
        }
    )
    return metrics


def summarize(rows):
    out = []
    grouped = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["depth"]), []).append(row)
    for (variant, depth), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: row["layer"])
        raw = [row["raw_bt_total_per_dim"] for row in group]
        post = [row["posthoc_bt_total_per_dim"] for row in group]
        raw_on = [row["raw_bt_on_diag_per_dim"] for row in group]
        post_on = [row["posthoc_bt_on_diag_per_dim"] for row in group]
        out.append(
            {
                "variant": variant,
                "depth": int(depth),
                "raw_first": float(raw[0]),
                "raw_last": float(raw[-1]),
                "raw_best": float(min(raw)),
                "raw_best_layer": int(group[raw.index(min(raw))]["layer"]),
                "posthoc_first": float(post[0]),
                "posthoc_last": float(post[-1]),
                "posthoc_best": float(min(post)),
                "posthoc_best_layer": int(group[post.index(min(post))]["layer"]),
                "raw_on_improvement": float(raw_on[0] - raw_on[-1]),
                "posthoc_on_improvement": float(post_on[0] - post_on[-1]),
                "posthoc_total_decrease_frac": float(
                    np.mean([(post[idx + 1] - post[idx]) <= 1e-4 for idx in range(len(post) - 1)])
                ),
            }
        )
    return out


def build_report(summaries):
    lines = [
        "# Post-Hoc Linear BT Diagnostic",
        "",
        "Fits one shared linear map on each frozen layer representation, applied to both views, to test whether BT information is present but mis-coordinate-aligned.",
        "",
        "| Variant | Depth | Raw first | Raw last | Raw best@ | Posthoc first | Posthoc last | Posthoc best@ | Posthoc dec frac | Raw on improvement | Posthoc on improvement |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['depth']} | "
            f"{row['raw_first']:.4f} | {row['raw_last']:.4f} | {row['raw_best']:.4f}@{row['raw_best_layer']} | "
            f"{row['posthoc_first']:.4f} | {row['posthoc_last']:.4f} | {row['posthoc_best']:.4f}@{row['posthoc_best_layer']} | "
            f"{row['posthoc_total_decrease_frac']:.2f} | {row['raw_on_improvement']:+.4f} | "
            f"{row['posthoc_on_improvement']:+.4f} |"
        )
    lines.append("")
    lines.append("Files: `posthoc_linear_rows.jsonl`, `posthoc_linear_summary.jsonl`.")
    return "\n".join(lines)


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="posthoc_bt_linear",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=args.num_classes,
            )
            for variant in args.variants:
                print(f"posthoc-linear depth={depth} seed={seed} variant={variant}", flush=True)
                state = collect_variant_state(point, variant, args, device, device_name)
                for layer_idx, (view1_np, view2_np) in enumerate(
                    zip(state["pathnorm_view1_train"], state["pathnorm_view2_train"])
                ):
                    view1 = torch.from_numpy(view1_np).to(device)
                    view2 = torch.from_numpy(view2_np).to(device)
                    raw = corr_metrics(view1, view2, args.bt_lambda)
                    row = {
                        "variant": variant,
                        "kind": state["kind"],
                        "depth": int(depth),
                        "seed": int(seed),
                        "layer": int(layer_idx + 1),
                    }
                    row.update({f"raw_{key}": value for key, value in raw.items()})
                    row.update(fit_shared_linear_bt(view1_np, view2_np, args, device))
                    rows.append(row)
                    write_jsonl(args.out_dir / "posthoc_linear_rows.partial.jsonl", rows)
                    del view1, view2
                del state
                torch.cuda.empty_cache()
                gc.collect()
    summaries = summarize(rows)
    write_jsonl(args.out_dir / "posthoc_linear_rows.jsonl", rows)
    write_jsonl(args.out_dir / "posthoc_linear_summary.jsonl", summaries)
    report = build_report(summaries)
    (args.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Post-hoc shared linear BT alignability diagnostic.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_posthoc_bt_linear_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.38)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[12])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--bt-offdiag-lambda", type=float, default=0.005)
    parser.add_argument("--fit-samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--postrelu-fit-samples", type=int, default=2048)
    parser.add_argument("--postrelu-steps", type=int, default=60)
    parser.add_argument("--postrelu-lr", type=float, default=0.05)
    parser.add_argument("--postrelu-scale-ridge", type=float, default=1e-4)
    parser.add_argument("--postrelu-bias-ridge", type=float, default=1e-4)
    parser.add_argument("--postrelu-grad-clip", type=float, default=10.0)
    parser.add_argument("--postnorm-linear-fit-samples", type=int, default=2048)
    parser.add_argument("--postnorm-linear-steps", type=int, default=60)
    parser.add_argument("--postnorm-linear-lr", type=float, default=0.03)
    parser.add_argument("--postnorm-linear-ridge", type=float, default=1e-4)
    parser.add_argument("--postnorm-linear-grad-clip", type=float, default=10.0)
    parser.add_argument("--postnorm-linear-cca-eps", type=float, default=1e-4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "plain_cf_postrelu_biasopt_relu",
            "plain_cf_agreement_biasopt_relu",
        ],
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
