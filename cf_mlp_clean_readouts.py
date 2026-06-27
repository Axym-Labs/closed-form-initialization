import argparse
import gc
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from cf_mlp_layer_mechanistic import class_separation_ratio, covariance_spectrum, view_alignment
from cf_mlp_representation import (
    apply_activation_torch,
    apply_map_np,
    one_hot_np,
    ridge_map_np,
    softmax_ce_np,
    standardize_pair,
    tensors_from_arrays,
)
from cf_mlp_scalability import SweepPoint, accuracy_from_logits, load_point_data, write_jsonl
from cf_mlp_scalability_gpu import (
    fit_cf_transform_torch,
    fit_whitening_transform_torch,
    normalize_hidden_with_stats_torch,
)


def invariance_schedule(name, depth):
    if name == "constant1":
        return [1.0] * depth
    if name == "relax2":
        return [float(0.5**idx) for idx in range(depth)]
    if name == "relax4":
        return [float(0.25**idx) for idx in range(depth)]
    if name.startswith("constant"):
        value = float(name.replace("constant", ""))
        return [value] * depth
    raise ValueError(f"Unknown invariance schedule: {name}")


def activation_config(name):
    if name == "relu":
        return "relu", 0.0
    if name == "gelu":
        return "gelu", 0.0
    if name == "identity":
        return "identity", 0.0
    if name.startswith("leakygelu"):
        return "leaky_gelu", float(name.replace("leakygelu", ""))
    if name.startswith("leaky"):
        return "leaky_relu", float(name.replace("leaky", ""))
    raise ValueError(f"Unknown activation: {name}")


def collect_depth_representations(point, device, device_name, transform_kind, schedule_name, activation_name):
    activation, activation_alpha = activation_config(activation_name)
    beta_schedule = invariance_schedule(schedule_name, point.depth)
    arrays = load_point_data(point)
    xtr_np, ytr_np, xte_np, yte_np, *_ = arrays
    tensors = tensors_from_arrays(arrays, device)
    xtr, _, xte, _, view1_tr, view2_tr, view1_te, view2_te = tensors

    train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
        [xtr, view1_tr, view2_tr],
        [xte, view1_te, view2_te],
    )
    base_tr, view1_tr, view2_tr = train_arrays
    base_te, view1_te, view2_te = test_arrays

    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1_train = []
    pathnorm_view2_train = []
    transform_rows = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for layer_idx in range(point.depth):
        if transform_kind == "cf":
            fitted = fit_cf_transform_torch(
                view1_tr,
                view2_tr,
                point.width,
                invariance_strength=beta_schedule[layer_idx],
            )
        elif transform_kind == "whiten":
            fitted = fit_whitening_transform_torch(view1_tr, view2_tr, point.width)
        else:
            raise ValueError(f"Unknown transform_kind: {transform_kind}")
        transform = fitted["transform"]

        base_tr = apply_activation_torch(base_tr @ transform, activation, activation_alpha)
        base_te = apply_activation_torch(base_te @ transform, activation, activation_alpha)
        view1_tr = apply_activation_torch(view1_tr @ transform, activation, activation_alpha)
        view2_tr = apply_activation_torch(view2_tr @ transform, activation, activation_alpha)
        view1_te = apply_activation_torch(view1_te @ transform, activation, activation_alpha)
        view2_te = apply_activation_torch(view2_te @ transform, activation, activation_alpha)

        train_arrays, test_arrays, _, _ = normalize_hidden_with_stats_torch(
            [base_tr, view1_tr, view2_tr],
            [base_te, view1_te, view2_te],
        )
        base_tr, view1_tr, view2_tr = train_arrays
        base_te, view1_te, view2_te = test_arrays

        pathnorm_train.append(base_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_test.append(base_te.detach().cpu().numpy().astype(np.float32))
        pathnorm_view1_train.append(view1_tr.detach().cpu().numpy().astype(np.float32))
        pathnorm_view2_train.append(view2_tr.detach().cpu().numpy().astype(np.float32))
        transform_rows.append(
            {
                "layer": layer_idx + 1,
                "transform_kind": transform_kind,
                "invariance_strength": None if transform_kind == "whiten" else beta_schedule[layer_idx],
                "old_lambda_reg": None if transform_kind == "whiten" else 1.0 / max(beta_schedule[layer_idx], 1e-12),
                "mean_gain": fitted["mean_gain"],
                "min_gain": fitted["min_gain"],
                "max_whitened_delta": fitted["max_whitened_delta"],
                "min_whitened_delta": fitted["min_whitened_delta"],
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_time = time.perf_counter() - start
    del tensors
    torch.cuda.empty_cache()
    return {
        "point": asdict(point),
        "device": device_name,
        "transform_kind": transform_kind,
        "schedule_name": schedule_name,
        "invariance_schedule": beta_schedule,
        "activation": activation,
        "activation_alpha": activation_alpha,
        "fit_time_sec": fit_time,
        "xtr": xtr_np.astype(np.float32),
        "xte": xte_np.astype(np.float32),
        "ytr": ytr_np.astype(np.int64),
        "yte": yte_np.astype(np.int64),
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
        "pathnorm_view1_train": pathnorm_view1_train,
        "pathnorm_view2_train": pathnorm_view2_train,
        "transform_rows": transform_rows,
    }


def linear_classifier_readout(xtr, xte, ytr, yte, reg):
    y_onehot = one_hot_np(ytr, int(np.max(ytr)) + 1)
    ztr, zte = standardize_pair(xtr, xte)
    weight = ridge_map_np(ztr, y_onehot, reg=reg, fit_bias=True)
    train_logits = apply_map_np(ztr, weight, fit_bias=True)
    test_logits = apply_map_np(zte, weight, fit_bias=True)
    return {
        "train_accuracy": accuracy_from_logits(train_logits, ytr),
        "test_accuracy": accuracy_from_logits(test_logits, yte),
        "train_ce": softmax_ce_np(train_logits, ytr),
        "test_ce": softmax_ce_np(test_logits, yte),
        "feature_dim": int(ztr.shape[1]),
        "readout_dim": int(ztr.shape[1]),
        "uses_pca": False,
    }


def pca512_readout(xtr, xte, ytr, yte, reg, seed, pca_dim):
    max_dim = min(pca_dim, xtr.shape[1], xtr.shape[0] - 1)
    pca = PCA(n_components=max_dim, svd_solver="randomized", iterated_power=3, random_state=seed)
    start = time.perf_counter()
    ztr = pca.fit_transform(xtr)
    zte = pca.transform(xte)
    pca_time = time.perf_counter() - start
    metrics = linear_classifier_readout(ztr.astype(np.float32), zte.astype(np.float32), ytr, yte, reg)
    metrics.update(
        {
            "feature_dim": int(xtr.shape[1]),
            "readout_dim": int(max_dim),
            "uses_pca": True,
            "pca_time_sec": float(pca_time),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_[:max_dim])),
        }
    )
    return metrics


def layer_quality_rows(state, variant, probe_reg):
    rows = []
    ytr = state["ytr"]
    yte = state["yte"]
    for idx, (xtr, xte, v1, v2) in enumerate(
        zip(
            state["pathnorm_train"],
            state["pathnorm_test"],
            state["pathnorm_view1_train"],
            state["pathnorm_view2_train"],
        )
    ):
        row = {
            "variant": variant,
            "seed": int(state["point"]["seed"]),
            "input_dim": int(state["point"]["input_dim"]),
            "width": int(state["point"]["width"]),
            "depth": int(state["point"]["depth"]),
            "transform_kind": state["transform_kind"],
            "schedule_name": state["schedule_name"],
            "activation": state["activation"],
            "activation_alpha": float(state["activation_alpha"]),
            "layer": idx + 1,
            "representation": "layer_hidden_512",
            "readout": "linear_on_layer_hidden_512",
            "supervised_mapping": "single_linear_classifier",
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, probe_reg))
        row.update(covariance_spectrum(xtr))
        row.update(class_separation_ratio(xtr, ytr))
        row.update(view_alignment(v1, v2, int(state["point"]["seed"]) + 1234 + idx))
        rows.append(row)
    return rows


def setup_rows(state, variant, probe_reg, pca_dim):
    ytr = state["ytr"]
    yte = state["yte"]
    depth = int(state["point"]["depth"])
    last_tr = state["pathnorm_train"][-1]
    last_te = state["pathnorm_test"][-1]
    all_tr = np.concatenate(state["pathnorm_train"], axis=1)
    all_te = np.concatenate(state["pathnorm_test"], axis=1)
    rows = []
    last = {
        "variant": variant,
        "seed": int(state["point"]["seed"]),
        "input_dim": int(state["point"]["input_dim"]),
        "width": int(state["point"]["width"]),
        "depth": depth,
        "transform_kind": state["transform_kind"],
        "schedule_name": state["schedule_name"],
        "activation": state["activation"],
        "activation_alpha": float(state["activation_alpha"]),
        "setup": "last_layer_512" if last_tr.shape[1] == pca_dim else "last_layer_pca512",
        "representation": "last_hidden_512" if last_tr.shape[1] == pca_dim else "last_hidden_to_pca512",
        "supervised_mapping": "single_linear_classifier",
    }
    if last_tr.shape[1] == pca_dim:
        last.update(linear_classifier_readout(last_tr, last_te, ytr, yte, probe_reg))
    else:
        last.update(pca512_readout(last_tr, last_te, ytr, yte, probe_reg, int(state["point"]["seed"]) + 17, pca_dim))
    rows.append(last)
    all_pca = {
        "variant": variant,
        "seed": int(state["point"]["seed"]),
        "input_dim": int(state["point"]["input_dim"]),
        "width": int(state["point"]["width"]),
        "depth": depth,
        "transform_kind": state["transform_kind"],
        "schedule_name": state["schedule_name"],
        "activation": state["activation"],
        "activation_alpha": float(state["activation_alpha"]),
        "setup": "all_layers_pca512",
        "representation": "all_hidden_concat_to_pca512",
        "supervised_mapping": "single_linear_classifier",
    }
    all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, probe_reg, int(state["point"]["seed"]), pca_dim))
    rows.append(all_pca)
    return rows


def aggregate(rows, key_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)
    numeric_keys = sorted(
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and key not in key_fields
    )
    out = []
    for key, items in sorted(grouped.items()):
        rec = {field: value for field, value in zip(key_fields, key)}
        rec["runs"] = len(items)
        for numeric_key in numeric_keys:
            vals = [float(item[numeric_key]) for item in items if numeric_key in item]
            if vals:
                rec[f"mean_{numeric_key}"] = float(np.mean(vals))
                rec[f"std_{numeric_key}"] = float(np.std(vals, ddof=0))
        out.append(rec)
    return out


def summarize_variant(state, variant, setup, layers):
    first_layer = next(row for row in layers if row["layer"] == 1)
    last_layer = next(row for row in layers if row["layer"] == int(state["point"]["depth"]))
    best_layer = max(layers, key=lambda row: row["test_accuracy"])
    setup_by_name = {row["setup"]: row for row in setup}
    last_setup = setup_by_name.get("last_layer_512", setup_by_name.get("last_layer_pca512"))
    return {
        "variant": variant,
        "seed": int(state["point"]["seed"]),
        "input_dim": int(state["point"]["input_dim"]),
        "width": int(state["point"]["width"]),
        "depth": int(state["point"]["depth"]),
        "transform_kind": state["transform_kind"],
        "schedule_name": state["schedule_name"],
        "invariance_schedule_json": json.dumps(
            [None if state["transform_kind"] == "whiten" else round(float(value), 10) for value in state["invariance_schedule"]]
        ),
        "activation": state["activation"],
        "activation_alpha": float(state["activation_alpha"]),
        "first_layer_accuracy": first_layer["test_accuracy"],
        "last_layer_accuracy": last_layer["test_accuracy"],
        "best_layer_accuracy": best_layer["test_accuracy"],
        "best_layer": int(best_layer["layer"]),
        "last_minus_first_accuracy": last_layer["test_accuracy"] - first_layer["test_accuracy"],
        "last_layer_effective_rank": last_layer["effective_rank"],
        "last_layer_view_mse_ratio": last_layer["same_over_shuffled_mse"],
        "last_layer_view_cosine": last_layer["same_view_cosine"],
        "last_setup": last_setup["setup"],
        "last_setup_accuracy": last_setup["test_accuracy"],
        "all_pca_setup_accuracy": setup_by_name["all_layers_pca512"]["test_accuracy"],
        "all_pca_explained_variance": setup_by_name["all_layers_pca512"]["explained_variance"],
        "fit_time_sec": float(state["fit_time_sec"]),
    }


def fmt(row, key):
    return f"{row.get(f'mean_{key}', float('nan')):.4f} +/- {row.get(f'std_{key}', float('nan')):.4f}"


def build_report(out_dir, summary_aggregate):
    lines = [
        "# Clean CF-MLP Readouts",
        "",
        "All supervised mappings here are a single linear classifier on a frozen representation.",
        "`last_layer_512` uses the final hidden activation directly when width is 512. `all_layers_pca512` concatenates all hidden activations and projects to 512 PCs before the same kind of linear classifier in the main runs.",
        "",
        "| Variant | Input dim | Width | Depth | Runs | Last layer setup | All layers PCA512 | Best layer | Last-first | Last rank | Last view ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_aggregate, key=lambda rec: (rec["input_dim"], rec["depth"], rec["variant"])):
        lines.append(
            f"| {row['variant']} | {row['input_dim']} | {row['width']} | {row['depth']} | {row['runs']} | "
            f"{fmt(row, 'last_setup_accuracy')} | "
            f"{fmt(row, 'all_pca_setup_accuracy')} | {fmt(row, 'best_layer_accuracy')} | "
            f"{fmt(row, 'last_minus_first_accuracy')} | "
            f"{row.get('mean_last_layer_effective_rank', float('nan')):.1f} +/- {row.get('std_last_layer_effective_rank', float('nan')):.1f} | "
            f"{row.get('mean_last_layer_view_mse_ratio', float('nan')):.3f} +/- {row.get('std_last_layer_view_mse_ratio', float('nan')):.3f} |"
        )
    lines.append("")
    report = "\n".join(lines)
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

    all_setup_rows = []
    all_layer_rows = []
    all_transform_rows = []
    all_summary_rows = []

    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="clean_readouts",
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
                transform_kind, schedule_name, activation_name = variant.split(":")
                print(
                    f"seed={seed} depth={depth} variant={variant} input_dim={args.input_dim} width={args.width} device={device_name}",
                    flush=True,
                )
                state = collect_depth_representations(
                    point,
                    device,
                    device_name,
                    transform_kind=transform_kind,
                    schedule_name=schedule_name,
                    activation_name=activation_name,
                )
                layer_rows = layer_quality_rows(state, variant, args.probe_reg)
                setup = setup_rows(state, variant, args.probe_reg, args.pca_dim)
                summary = summarize_variant(state, variant, setup, layer_rows)
                all_layer_rows.extend(layer_rows)
                all_setup_rows.extend(setup)
                all_transform_rows.extend({"variant": variant, "seed": seed, "depth": depth, **row} for row in state["transform_rows"])
                all_summary_rows.append(summary)
                write_jsonl(args.out_dir / "layer_readouts.partial.jsonl", all_layer_rows)
                write_jsonl(args.out_dir / "setup_readouts.partial.jsonl", all_setup_rows)
                write_jsonl(args.out_dir / "transform_rows.partial.jsonl", all_transform_rows)
                write_jsonl(args.out_dir / "summary.partial.jsonl", all_summary_rows)
                del state, layer_rows, setup, summary
                torch.cuda.empty_cache()
                gc.collect()

    summary_aggregate = aggregate(
        all_summary_rows,
        ["variant", "transform_kind", "schedule_name", "activation", "activation_alpha", "input_dim", "width", "depth"],
    )
    layer_aggregate = aggregate(all_layer_rows, ["variant", "input_dim", "depth", "layer"])
    setup_aggregate = aggregate(all_setup_rows, ["variant", "input_dim", "depth", "setup"])
    write_jsonl(args.out_dir / "layer_readouts.jsonl", all_layer_rows)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", all_setup_rows)
    write_jsonl(args.out_dir / "transform_rows.jsonl", all_transform_rows)
    write_jsonl(args.out_dir / "summary.jsonl", all_summary_rows)
    write_jsonl(args.out_dir / "summary_aggregate.jsonl", summary_aggregate)
    write_jsonl(args.out_dir / "layer_readouts_aggregate.jsonl", layer_aggregate)
    write_jsonl(args.out_dir / "setup_readouts_aggregate.jsonl", setup_aggregate)
    print(build_report(args.out_dir, summary_aggregate), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Clean frozen-representation readouts for CF-MLP depth paths.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_clean_readouts"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_fullres_width")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "cf:relax4:relu",
            "whiten:relax4:relu",
            "cf:relax4:leaky0.2",
            "cf:relax4:leakygelu0.5",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
