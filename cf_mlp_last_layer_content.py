import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cf_mlp_clean_readouts import collect_depth_representations, linear_classifier_readout
from cf_mlp_layer_mechanistic import covariance_spectrum, view_alignment
from cf_mlp_representation import apply_map_np, ridge_map_np, standardize_pair
from cf_mlp_scalability import SweepPoint, write_jsonl


def center(x):
    return x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)


def linear_cka(x, y):
    x0 = center(x)
    y0 = center(y)
    xy = x0.T @ y0
    xx = x0.T @ x0
    yy = y0.T @ y0
    num = float(np.sum(xy * xy))
    den = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return num / max(den, 1e-12)


def class_one_hot(y, num_classes):
    eye = np.eye(num_classes, dtype=np.float32)
    return eye[y]


def raw_reconstruction_metrics(htr, hte, xtr, xte, reg):
    htr_z, hte_z = standardize_pair(htr, hte)
    xtr_z, xte_z = standardize_pair(xtr, xte)
    weight = ridge_map_np(htr_z, xtr_z, reg=reg, fit_bias=True)
    pred = apply_map_np(hte_z, weight, fit_bias=True)
    mse = float(np.mean((pred - xte_z) ** 2))
    baseline = float(np.mean((xte_z - xtr_z.mean(axis=0, keepdims=True)) ** 2))
    return {
        "raw_reconstruction_r2": float(1.0 - mse / max(baseline, 1e-12)),
        "raw_reconstruction_mse": mse,
    }


def normalize_rows(x):
    x0 = x.astype(np.float32)
    x0 = x0 - x0.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x0, axis=1, keepdims=True)
    return x0 / np.maximum(norm, 1e-12)


def view_retrieval_metrics(v1, v2, max_items):
    n = min(max_items, v1.shape[0], v2.shape[0])
    a = normalize_rows(v1[:n])
    b = normalize_rows(v2[:n])
    sims = a @ b.T
    ranks = np.argsort(-sims, axis=1)
    target = np.arange(n)
    top1 = float(np.mean(ranks[:, 0] == target))
    top5 = float(np.mean([target[idx] in ranks[idx, :5] for idx in range(n)]))
    pos = sims[target, target]
    max_neg = np.max(sims + np.eye(n, dtype=np.float32) * -2.0, axis=1)
    return {
        "view_retrieval_n": n,
        "view_retrieval_top1": top1,
        "view_retrieval_top5": top5,
        "view_positive_similarity": float(np.mean(pos)),
        "view_max_negative_similarity": float(np.mean(max_neg)),
        "view_positive_minus_max_negative": float(np.mean(pos - max_neg)),
    }


def class_knn_metrics(h, y, max_items, k):
    n = min(max_items, h.shape[0], y.shape[0])
    z = normalize_rows(h[:n])
    labels = y[:n]
    sims = z @ z.T
    np.fill_diagonal(sims, -2.0)
    nn = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    purity = float(np.mean(labels[nn] == labels[:, None]))
    return {
        "class_knn_n": n,
        "class_knn_k": k,
        "class_knn_purity": purity,
    }


def depth_points(depth):
    mids = sorted(set([1, max(1, depth // 2), depth]))
    return [idx - 1 for idx in mids]


def analyze_state(state, variant, probe_reg, recon_reg, max_retrieval, knn_k):
    ytr = state["ytr"]
    yte = state["yte"]
    num_classes = int(state["point"]["num_classes"])
    first = state["pathnorm_train"][0]
    rows = []
    for idx in depth_points(len(state["pathnorm_train"])):
        htr = state["pathnorm_train"][idx]
        hte = state["pathnorm_test"][idx]
        v1 = state["pathnorm_view1_train"][idx]
        v2 = state["pathnorm_view2_train"][idx]
        row = {
            "variant": variant,
            "seed": int(state["point"]["seed"]),
            "input_dim": int(state["point"]["input_dim"]),
            "width": int(state["point"]["width"]),
            "depth": int(state["point"]["depth"]),
            "layer": idx + 1,
            "is_last_layer": bool(idx == len(state["pathnorm_train"]) - 1),
            "activation": state["activation"],
            "activation_alpha": float(state["activation_alpha"]),
            "schedule_name": state["schedule_name"],
            "transform_kind": state["transform_kind"],
            "invariance_strength": None if state["transform_kind"] == "whiten" else state["invariance_schedule"][idx],
        }
        readout = linear_classifier_readout(htr, hte, ytr, yte, probe_reg)
        row["class_linear_accuracy"] = readout["test_accuracy"]
        row["class_linear_train_accuracy"] = readout["train_accuracy"]
        row.update(covariance_spectrum(htr))
        row.update(view_alignment(v1, v2, int(state["point"]["seed"]) + idx + 4321))
        row.update(raw_reconstruction_metrics(htr, hte, state["xtr"], state["xte"], recon_reg))
        row.update(view_retrieval_metrics(v1, v2, max_retrieval))
        row.update(class_knn_metrics(hte, yte, max_retrieval, knn_k))
        row["cka_to_raw_input"] = linear_cka(htr, state["xtr"])
        row["cka_to_labels"] = linear_cka(htr, class_one_hot(ytr, num_classes))
        row["cka_to_first_layer"] = linear_cka(htr, first)
        rows.append(row)
    return rows


def aggregate(rows, key_fields):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in key_fields), []).append(row)
    out = []
    for key, items in sorted(grouped.items()):
        rec = {field: value for field, value in zip(key_fields, key)}
        rec["runs"] = len(items)
        numeric = sorted(
            key
            for row in items
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and key not in key_fields
        )
        for name in numeric:
            vals = [float(item[name]) for item in items if name in item and item[name] is not None]
            if vals:
                rec[f"mean_{name}"] = float(np.mean(vals))
                rec[f"std_{name}"] = float(np.std(vals, ddof=0))
        out.append(rec)
    return out


def build_report(out_dir, aggregate_rows):
    lines = [
        "# Last-Layer Representation Content",
        "",
        "Rows are frozen hidden representations evaluated with one linear classifier or unsupervised content metrics. No residual supervised heads are used.",
        "",
        "| Variant | Depth | Layer | Class acc | Raw recon R2 | CKA raw | CKA labels | CKA first | View top1 | kNN class | Rank | View ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| {row['variant']} | {row['depth']} | {row['layer']} | "
            f"{row.get('mean_class_linear_accuracy', float('nan')):.4f} | "
            f"{row.get('mean_raw_reconstruction_r2', float('nan')):.3f} | "
            f"{row.get('mean_cka_to_raw_input', float('nan')):.3f} | "
            f"{row.get('mean_cka_to_labels', float('nan')):.3f} | "
            f"{row.get('mean_cka_to_first_layer', float('nan')):.3f} | "
            f"{row.get('mean_view_retrieval_top1', float('nan')):.3f} | "
            f"{row.get('mean_class_knn_purity', float('nan')):.3f} | "
            f"{row.get('mean_effective_rank', float('nan')):.1f} | "
            f"{row.get('mean_same_over_shuffled_mse', float('nan')):.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset="cifar100_fullres_width",
                axis="last_layer_content",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=100,
            )
            for variant in args.variants:
                transform_kind, schedule_name, activation_name = variant.split(":")
                print(f"seed={seed} depth={depth} variant={variant}", flush=True)
                state = collect_depth_representations(
                    point,
                    device,
                    device_name,
                    transform_kind=transform_kind,
                    schedule_name=schedule_name,
                    activation_name=activation_name,
                )
                rows = analyze_state(state, variant, args.probe_reg, args.recon_reg, args.max_retrieval, args.knn_k)
                all_rows.extend(rows)
                write_jsonl(args.out_dir / "content_rows.partial.jsonl", all_rows)
                del state
                torch.cuda.empty_cache()
    agg = aggregate(all_rows, ["variant", "input_dim", "width", "depth", "layer"])
    write_jsonl(args.out_dir / "content_rows.jsonl", all_rows)
    write_jsonl(args.out_dir / "content_aggregate.jsonl", agg)
    print(build_report(args.out_dir, agg), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Analyze what depth-scaled CF last-layer representations encode.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_last_layer_content"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--recon-reg", type=float, default=100.0)
    parser.add_argument("--max-retrieval", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=["cf:relax4:leaky0.2", "cf:relax4:leakygelu0.5"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
