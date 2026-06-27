import argparse
import csv
import gc
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cf_mlp_barlow_clean import collect_barlow_representations, load_tensors
from cf_mlp_bt_objective_by_layer import find_nonres_bt_model, find_residual_bt_model, residual_cf_args
from cf_mlp_last_layer_content import linear_cka
from cf_mlp_residual_barlow import collect_residual_representations
from cf_mlp_residual_bt_variants import collect_variant_state
from cf_mlp_scalability import SweepPoint, write_jsonl


def standardize(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-6)


def subsample(x, max_samples):
    if max_samples <= 0 or x.shape[0] <= max_samples:
        return x
    idx = np.linspace(0, x.shape[0] - 1, max_samples).astype(np.int64)
    return x[idx]


def ridge_predict_r2(x, y, reg):
    xz = standardize(x)
    yz = standardize(y)
    gram = xz.T @ xz
    rhs = xz.T @ yz
    eye = np.eye(gram.shape[0], dtype=np.float64)
    weight = np.linalg.solve(gram + float(reg) * xz.shape[0] * eye, rhs)
    pred = xz @ weight
    mse = float(np.mean((pred - yz) ** 2))
    baseline = float(np.mean(yz * yz))
    return float(1.0 - mse / max(baseline, 1e-12)), mse


def transition_metrics(prev, cur, reg, max_samples):
    prev = subsample(prev, max_samples)
    cur = subsample(cur, max_samples)
    cka = linear_cka(prev, cur)
    forward_r2, forward_mse = ridge_predict_r2(prev, cur, reg)
    reverse_r2, reverse_mse = ridge_predict_r2(cur, prev, reg)
    sym_r2 = 0.5 * (forward_r2 + reverse_r2)
    return {
        "adjacent_cka": float(cka),
        "forward_linear_r2": float(forward_r2),
        "reverse_linear_r2": float(reverse_r2),
        "sym_linear_r2": float(sym_r2),
        "linear_novelty": float(1.0 - sym_r2),
        "forward_linear_mse": float(forward_mse),
        "reverse_linear_mse": float(reverse_mse),
    }


def layer_metrics(first, cur, raw_input, label_onehot, max_samples):
    first_s = subsample(first, max_samples)
    cur_s = subsample(cur, max_samples)
    raw_s = subsample(raw_input, max_samples)
    label_s = subsample(label_onehot, max_samples)
    return {
        "cka_to_first": float(linear_cka(cur_s, first_s)),
        "cka_to_raw_input": float(linear_cka(cur_s, raw_s)),
        "cka_to_labels": float(linear_cka(cur_s, label_s)),
    }


def class_onehot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[np.asarray(y, dtype=np.int64)]


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="layer_geometry_drift",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def collect_model_reps(args, point, model_name, variant, device, device_name):
    tensors = load_tensors(point, device)
    if model_name == "residual_backprop_bt":
        model_path = find_residual_bt_model(args.bt_model_dir, point.depth)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        reps = collect_residual_representations(point, tensors, state, device)
        checkpoint = str(model_path)
    elif model_name == "nonres_backprop_bt":
        model_path = find_nonres_bt_model(args.nonres_bt_model_dir, point.depth)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        reps = collect_barlow_representations(point, tensors, state, device)
        checkpoint = str(model_path)
    elif model_name == "cf_bt":
        state = collect_variant_state(point, variant, residual_cf_args(args), device, device_name)
        reps = {
            "pathnorm_train": state["pathnorm_train"],
            "pathnorm_view1_train": state["pathnorm_view1_train"],
            "pathnorm_view2_train": state["pathnorm_view2_train"],
        }
        checkpoint = ""
        del state
    else:
        raise ValueError(f"Unknown model: {model_name}")
    xtr = tensors["xtr_np"].astype(np.float32)
    ytr = tensors["ytr_np"].astype(np.int64)
    del tensors
    torch.cuda.empty_cache()
    gc.collect()
    return reps, xtr, ytr, checkpoint


def rows_for_reps(args, point, model_name, variant, reps, xtr, ytr, checkpoint):
    label_onehot = class_onehot(ytr, point.num_classes)
    streams = {
        "base": reps["pathnorm_train"],
        "view1": reps["pathnorm_view1_train"],
        "view2": reps["pathnorm_view2_train"],
    }
    rows = []
    for stream_name, layers in streams.items():
        first = layers[0]
        for idx, cur in enumerate(layers):
            row = {
                "model": model_name,
                "variant": variant,
                "stream": stream_name,
                "dataset": point.dataset,
                "seed": point.seed,
                "depth": point.depth,
                "layer": idx + 1,
                "width": point.width,
                "input_dim": point.input_dim,
                "checkpoint": checkpoint,
            }
            row.update(layer_metrics(first, cur, xtr, label_onehot, args.max_metric_samples))
            if idx > 0:
                row.update(transition_metrics(layers[idx - 1], cur, args.ridge_reg, args.max_metric_samples))
            else:
                row.update(
                    {
                        "adjacent_cka": float("nan"),
                        "forward_linear_r2": float("nan"),
                        "reverse_linear_r2": float("nan"),
                        "sym_linear_r2": float("nan"),
                        "linear_novelty": float("nan"),
                        "forward_linear_mse": float("nan"),
                        "reverse_linear_mse": float("nan"),
                    }
                )
            rows.append(row)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = (row["model"], row["variant"], row["stream"], row["depth"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for (model, variant, stream, depth), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["layer"])
        trans = [row for row in ordered if row["layer"] > 1]
        final = ordered[-1]
        mean_adj_cka = float(np.nanmean([row["adjacent_cka"] for row in trans])) if trans else float("nan")
        mean_sym_r2 = float(np.nanmean([row["sym_linear_r2"] for row in trans])) if trans else float("nan")
        mean_novelty = float(np.nanmean([row["linear_novelty"] for row in trans])) if trans else float("nan")
        summaries.append(
            {
                "model": model,
                "variant": variant,
                "stream": stream,
                "depth": depth,
                "final_cka_to_first": final["cka_to_first"],
                "final_cka_to_raw_input": final["cka_to_raw_input"],
                "final_cka_to_labels": final["cka_to_labels"],
                "mean_adjacent_cka": mean_adj_cka,
                "mean_sym_linear_r2": mean_sym_r2,
                "mean_linear_novelty": mean_novelty,
                "final_sym_linear_r2": final["sym_linear_r2"],
                "final_linear_novelty": final["linear_novelty"],
            }
        )
    return summaries


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(path, summaries):
    lines = [
        "# Layer Geometry Drift",
        "",
        "Linear predictability and CKA are computed on path-normalized hidden states. Low novelty means the next layer is close to a linear reparameterization of the previous layer.",
        "",
        "| Model | Variant | Stream | Depth | Final CKA to L1 | Mean adjacent CKA | Mean sym linear R2 | Mean novelty | Final novelty | CKA labels |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    row["variant"],
                    row["stream"],
                    str(row["depth"]),
                    fmt(row["final_cka_to_first"]),
                    fmt(row["mean_adjacent_cka"]),
                    fmt(row["mean_sym_linear_r2"]),
                    fmt(row["mean_linear_novelty"]),
                    fmt(row["final_linear_novelty"]),
                    fmt(row["final_cka_to_labels"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Files: `layer_geometry_rows.csv`, `layer_geometry_summary.csv`.")
    path.write_text("\n".join(lines), encoding="utf-8")


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
            point = point_for(args, depth, seed)
            for model_name in args.models:
                variants = args.cf_variants if model_name == "cf_bt" else [""]
                for variant in variants:
                    print(f"geometry depth={depth} seed={seed} model={model_name} variant={variant}", flush=True)
                    reps, xtr, ytr, checkpoint = collect_model_reps(args, point, model_name, variant, device, device_name)
                    rows.extend(rows_for_reps(args, point, model_name, variant, reps, xtr, ytr, checkpoint))
                    write_jsonl(args.out_dir / "layer_geometry_rows.partial.jsonl", rows)
                    del reps, xtr, ytr
                    gc.collect()
    summaries = summarize(rows)
    write_jsonl(args.out_dir / "layer_geometry_rows.jsonl", rows)
    write_jsonl(args.out_dir / "layer_geometry_summary.jsonl", summaries)
    write_csv(args.out_dir / "layer_geometry_rows.csv", rows)
    write_csv(args.out_dir / "layer_geometry_summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Layerwise representation drift diagnostics for CF-BT and BP-BT.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_layer_geometry_drift_seed7"))
    parser.add_argument("--bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7"))
    parser.add_argument("--nonres-bt-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_barlow_layer_only_cifar100_simclr_seed7"))
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
    parser.add_argument("--cf-residual-scale", type=float, default=1.0)
    parser.add_argument("--models", nargs="+", default=["residual_backprop_bt", "nonres_backprop_bt", "cf_bt"])
    parser.add_argument(
        "--cf-variants",
        nargs="+",
        default=[
            "plain_cf_relu",
            "plain_cf_agreement_biasopt_relu",
            "plain_cf_agreement_biasopt_ccalinear_relu",
            "plain_cf_agreement_biasopt_linearopt_relu",
        ],
    )
    parser.add_argument("--ridge-reg", type=float, default=1e-3)
    parser.add_argument("--max-metric-samples", type=int, default=12000)
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
