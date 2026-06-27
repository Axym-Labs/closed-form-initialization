import argparse
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_clean_readouts import (
    linear_classifier_readout,
    pca512_readout,
)
from cf_mlp_last_layer_content import (
    class_knn_metrics,
    class_one_hot,
    linear_cka,
    raw_reconstruction_metrics,
    view_retrieval_metrics,
)
from cf_mlp_layer_mechanistic import covariance_spectrum, standardize_many, view_alignment
from cf_mlp_scalability import SweepPoint, load_point_data, write_jsonl
from cf_mlp_scalability_gpu import init_backprop_params, normalize_hidden_with_stats_torch


def forward_hiddens(x, weights):
    h = x
    hiddens = []
    for weight in weights:
        h = torch.relu(h @ weight)
        hiddens.append(h)
    return hiddens


def off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(z1, z2, lambd):
    z1 = (z1 - z1.mean(dim=0)) / torch.clamp(z1.std(dim=0), min=1e-4)
    z2 = (z2 - z2.mean(dim=0)) / torch.clamp(z2.std(dim=0), min=1e-4)
    corr = (z1.T @ z2) / z1.shape[0]
    on_diag = torch.diagonal(corr).add(-1.0).pow(2).sum()
    off_diag = off_diagonal(corr).pow(2).sum()
    return on_diag + float(lambd) * off_diag


def parse_config(text):
    parts = text.split(":")
    if len(parts) != 7:
        raise ValueError(
            "Config format is name:epochs:lr:batch_size:projector_dim:bt_lambda:weight_decay"
        )
    return {
        "name": parts[0],
        "epochs": float(parts[1]),
        "lr": float(parts[2]),
        "batch_size": int(parts[3]),
        "projector_dim": int(parts[4]),
        "bt_lambda": float(parts[5]),
        "weight_decay": float(parts[6]),
    }


def config_hash(config):
    text = json.dumps(config, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def load_tensors(point, device):
    arrays = load_point_data(point)
    xtr_np, ytr_np, xte_np, yte_np, *_ = arrays

    def x_tensor(arr):
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)

    xtr, ytr, xte, yte, view1_tr, view2_tr, view1_te, view2_te = (
        x_tensor(arrays[0]),
        torch.from_numpy(arrays[1].astype(np.int64)).to(device),
        x_tensor(arrays[2]),
        torch.from_numpy(arrays[3].astype(np.int64)).to(device),
        x_tensor(arrays[4]),
        x_tensor(arrays[5]),
        x_tensor(arrays[6]),
        x_tensor(arrays[7]),
    )
    return {
        "arrays": arrays,
        "xtr_np": xtr_np.astype(np.float32),
        "xte_np": xte_np.astype(np.float32),
        "ytr_np": ytr_np.astype(np.int64),
        "yte_np": yte_np.astype(np.int64),
        "xtr": xtr,
        "ytr": ytr,
        "xte": xte,
        "yte": yte,
        "view1_tr": view1_tr,
        "view2_tr": view2_tr,
        "view1_te": view1_te,
        "view2_te": view2_te,
    }


def normalized_inputs(tensors, norm_mean=None, norm_scale=None):
    if norm_mean is None or norm_scale is None:
        _, _, norm_mean, norm_scale = normalize_hidden_with_stats_torch(
            [tensors["xtr"], tensors["view1_tr"], tensors["view2_tr"]],
            [tensors["xte"], tensors["view1_te"], tensors["view2_te"]],
        )
    return {
        "norm_mean": norm_mean,
        "norm_scale": norm_scale,
        "xtr": (tensors["xtr"] - norm_mean) / norm_scale,
        "xte": (tensors["xte"] - norm_mean) / norm_scale,
        "view1_tr": (tensors["view1_tr"] - norm_mean) / norm_scale,
        "view2_tr": (tensors["view2_tr"] - norm_mean) / norm_scale,
        "view1_te": (tensors["view1_te"] - norm_mean) / norm_scale,
        "view2_te": (tensors["view2_te"] - norm_mean) / norm_scale,
    }


def train_barlow(point, tensors, device, config, print_every_epochs):
    tuned_point = replace(point, lr=config["lr"], weight_decay=config["weight_decay"])
    normed = normalized_inputs(tensors)
    weights, _ = init_backprop_params(tuned_point, device)
    final_dim = min(point.width, point.input_dim)
    torch.manual_seed(point.seed + 707)
    projector = torch.nn.Parameter(
        torch.randn((final_dim, config["projector_dim"]), device=device) * math.sqrt(2.0 / final_dim)
    )
    params = list(weights.parameters()) + [projector]
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    steps_per_epoch = max(1, int(math.ceil(point.n_train / config["batch_size"])))
    total_steps = int(math.ceil(config["epochs"] * steps_per_epoch))
    permutation = torch.randperm(point.n_train, device=device)
    cursor = 0
    losses = []
    history = []
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()
    for step in range(1, total_steps + 1):
        if cursor + config["batch_size"] > point.n_train:
            permutation = torch.randperm(point.n_train, device=device)
            cursor = 0
        batch_idx = permutation[cursor : cursor + config["batch_size"]]
        cursor += config["batch_size"]
        h1 = forward_hiddens(normed["view1_tr"][batch_idx], weights)[-1]
        h2 = forward_hiddens(normed["view2_tr"][batch_idx], weights)[-1]
        loss = barlow_loss(h1 @ projector, h2 @ projector, config["bt_lambda"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step % steps_per_epoch == 0 or step == total_steps:
            epoch = step / steps_per_epoch
            recent = float(np.mean(losses[-min(len(losses), steps_per_epoch) :]))
            if epoch % print_every_epochs < 1e-9 or step == total_steps:
                print(f"bt {config['name']} epoch={epoch:.1f} loss={recent:.3f}", flush=True)
            history.append({"step": step, "epoch": epoch, "mean_barlow_loss": recent})
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "weights": [param.detach().cpu() for param in weights],
        "projector": projector.detach().cpu(),
        "norm_mean": normed["norm_mean"].detach().cpu(),
        "norm_scale": normed["norm_scale"].detach().cpu(),
        "history": history,
        "fit_time_sec": elapsed,
        "config": dict(config),
    }


def load_saved_barlow(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "weights": state["weights"],
        "projector": state["projector"],
        "norm_mean": state["norm_mean"],
        "norm_scale": state["norm_scale"],
        "history": state.get("history", []),
        "fit_time_sec": float(state.get("fit_time_sec", 0.0)),
        "config": {
            "name": path.stem,
            "epochs": float(state.get("effective_epochs", 0.0)),
            "lr": float(state["point"].get("lr", 0.0)) if isinstance(state.get("point"), dict) else 0.0,
            "batch_size": int(state.get("bt_batch_size", 0)),
            "projector_dim": int(state["projector"].shape[1]),
            "bt_lambda": float(state.get("bt_lambda", 0.0)),
            "weight_decay": float(state["point"].get("weight_decay", 0.0)) if isinstance(state.get("point"), dict) else 0.0,
        },
    }


def collect_barlow_representations(point, tensors, state, device):
    norm_mean = state["norm_mean"].to(device)
    norm_scale = state["norm_scale"].to(device)
    normed = normalized_inputs(tensors, norm_mean, norm_scale)
    weights = [param.to(device) for param in state["weights"]]
    with torch.no_grad():
        train_h = forward_hiddens(normed["xtr"], weights)
        test_h = forward_hiddens(normed["xte"], weights)
        view1_h = forward_hiddens(normed["view1_tr"], weights)
        view2_h = forward_hiddens(normed["view2_tr"], weights)
    raw_train = [h.detach().cpu().numpy().astype(np.float32) for h in train_h]
    raw_test = [h.detach().cpu().numpy().astype(np.float32) for h in test_h]
    raw_view1 = [h.detach().cpu().numpy().astype(np.float32) for h in view1_h]
    raw_view2 = [h.detach().cpu().numpy().astype(np.float32) for h in view2_h]
    pathnorm_train = []
    pathnorm_test = []
    pathnorm_view1 = []
    pathnorm_view2 = []
    for htr, hte, hv1, hv2 in zip(raw_train, raw_test, raw_view1, raw_view2):
        htr_s, hte_s, hv1_s, hv2_s = standardize_many(htr, hte, hv1, hv2)
        pathnorm_train.append(htr_s)
        pathnorm_test.append(hte_s)
        pathnorm_view1.append(hv1_s)
        pathnorm_view2.append(hv2_s)
    return {
        "raw_train": raw_train,
        "raw_test": raw_test,
        "raw_view1_train": raw_view1,
        "raw_view2_train": raw_view2,
        "pathnorm_train": pathnorm_train,
        "pathnorm_test": pathnorm_test,
        "pathnorm_view1_train": pathnorm_view1,
        "pathnorm_view2_train": pathnorm_view2,
    }


def readout_rows(point, model_name, config, reps, ytr, yte, probe_reg, pca_dim, include_pca=True):
    rows = []
    for idx, (xtr, xte, v1, v2) in enumerate(
        zip(
            reps["pathnorm_train"],
            reps["pathnorm_test"],
            reps["pathnorm_view1_train"],
            reps["pathnorm_view2_train"],
        )
    ):
        row = {
            "model": model_name,
            "config": config["name"],
            "seed": point.seed,
            "input_dim": point.input_dim,
            "width": point.width,
            "depth": point.depth,
            "layer": idx + 1,
            "setup": "layer_hidden_512",
            "representation": "bt_layer_hidden_512",
            "supervised_mapping": "single_linear_classifier",
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, probe_reg))
        row.update(covariance_spectrum(xtr))
        row.update(view_alignment(v1, v2, point.seed + idx + 8181))
        rows.append(row)

    last_tr = reps["pathnorm_train"][-1]
    last_te = reps["pathnorm_test"][-1]
    last = {
        "model": model_name,
        "config": config["name"],
        "seed": point.seed,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "setup": "last_layer_512",
        "representation": "bt_last_hidden_512",
        "supervised_mapping": "single_linear_classifier",
    }
    last.update(linear_classifier_readout(last_tr, last_te, ytr, yte, probe_reg))
    setup_rows = [last]
    if include_pca:
        all_tr = np.concatenate(reps["pathnorm_train"], axis=1)
        all_te = np.concatenate(reps["pathnorm_test"], axis=1)
        all_pca = {
            "model": model_name,
            "config": config["name"],
            "seed": point.seed,
            "input_dim": point.input_dim,
            "width": point.width,
            "depth": point.depth,
            "setup": "all_layers_pca512",
            "representation": "bt_all_hidden_concat_to_pca512",
            "supervised_mapping": "single_linear_classifier",
        }
        all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, probe_reg, point.seed, pca_dim))
        setup_rows.append(all_pca)
    return rows, setup_rows


def content_rows(point, model_name, config, reps, xtr, xte, ytr, yte, probe_reg, recon_reg, pca_dim, max_retrieval, knn_k):
    rows = []
    features = [
        (
            "last_layer_512",
            "bt_last_hidden_512",
            reps["pathnorm_train"][-1],
            reps["pathnorm_test"][-1],
            reps["pathnorm_view1_train"][-1],
            reps["pathnorm_view2_train"][-1],
        ),
    ]
    all_tr = np.concatenate(reps["pathnorm_train"], axis=1)
    all_te = np.concatenate(reps["pathnorm_test"], axis=1)
    all_v1 = np.concatenate(reps["pathnorm_view1_train"], axis=1)
    all_v2 = np.concatenate(reps["pathnorm_view2_train"], axis=1)
    pca_seed = point.seed + 991
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(pca_dim, all_tr.shape[1]), svd_solver="randomized", iterated_power=3, random_state=pca_seed)
    all_tr_pca = pca.fit_transform(all_tr).astype(np.float32)
    all_te_pca = pca.transform(all_te).astype(np.float32)
    all_v1_pca = pca.transform(all_v1).astype(np.float32)
    all_v2_pca = pca.transform(all_v2).astype(np.float32)
    features.append(("all_layers_pca512", "bt_all_hidden_concat_to_pca512", all_tr_pca, all_te_pca, all_v1_pca, all_v2_pca))

    labels_onehot = class_one_hot(ytr, point.num_classes)
    for setup, representation, ftr, fte, v1, v2 in features:
        readout = linear_classifier_readout(ftr, fte, ytr, yte, probe_reg)
        row = {
            "model": model_name,
            "config": config["name"],
            "seed": point.seed,
            "input_dim": point.input_dim,
            "width": point.width,
            "depth": point.depth,
            "setup": setup,
            "representation": representation,
            "class_linear_accuracy": readout["test_accuracy"],
            "class_linear_train_accuracy": readout["train_accuracy"],
            "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)) if setup == "all_layers_pca512" else 1.0,
        }
        row.update(covariance_spectrum(ftr))
        row.update(view_alignment(v1, v2, point.seed + 9191))
        row.update(raw_reconstruction_metrics(ftr, fte, xtr, xte, recon_reg))
        row.update(view_retrieval_metrics(v1, v2, max_retrieval))
        row.update(class_knn_metrics(fte, yte, max_retrieval, knn_k))
        row["cka_to_raw_input"] = linear_cka(ftr, xtr)
        row["cka_to_labels"] = linear_cka(ftr, labels_onehot)
        row["cka_to_first_layer"] = linear_cka(ftr, reps["pathnorm_train"][0])
        rows.append(row)
    return rows


def summarize_setup(point, model_name, config, setup_rows):
    by_setup = {row["setup"]: row for row in setup_rows}
    best_layer = max((row for row in setup_rows if row["setup"] == "layer_hidden_512"), key=lambda row: row["test_accuracy"])
    last_layer = next(row for row in setup_rows if row["setup"] == "layer_hidden_512" and row["layer"] == point.depth)
    all_pca = by_setup.get("all_layers_pca512", {})
    return {
        "model": model_name,
        "config": config["name"],
        "seed": point.seed,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "epochs": float(config.get("epochs", 0.0)),
        "lr": float(config.get("lr", 0.0)),
        "batch_size": int(config.get("batch_size", 0)),
        "projector_dim": int(config.get("projector_dim", 0)),
        "bt_lambda": float(config.get("bt_lambda", 0.0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "last_layer_accuracy": by_setup.get("last_layer_512", last_layer)["test_accuracy"],
        "all_pca_accuracy": all_pca.get("test_accuracy", float("nan")),
        "best_layer_accuracy": best_layer["test_accuracy"],
        "best_layer": int(best_layer["layer"]),
        "last_layer_effective_rank": by_setup.get("last_layer_512", last_layer).get("effective_rank", float("nan")),
        "all_pca_explained_variance": all_pca.get("explained_variance", float("nan")),
    }


def build_report(out_dir, summary_rows, content):
    def value_or_na(value):
        if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4f}"

    lines = [
        "# Clean Barlow Twins Readouts",
        "",
        "Positive pairs are same-data-instance augmentations under the selected dataset policy.",
        "All supervised mappings are a single linear classifier on a frozen representation.",
        "",
        "| Config | Epochs | Projector | BT lambda | Last 512 | All PCA512 | Best layer | Best layer idx |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summary_rows, key=lambda item: item["config"]):
        lines.append(
            f"| {row['config']} | {row['epochs']:.0f} | {row['projector_dim']} | {row['bt_lambda']:.4g} | "
            f"{row['last_layer_accuracy']:.4f} | {value_or_na(row['all_pca_accuracy'])} | "
            f"{row['best_layer_accuracy']:.4f} | {row['best_layer']} |"
        )
    if content:
        lines.extend(
            [
                "",
                "## Representation Content",
                "",
                "| Config | Setup | Class acc | Raw recon R2 | CKA raw | CKA labels | View top1 | kNN class | Rank | View ratio |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(content, key=lambda item: (item["config"], item["setup"])):
            lines.append(
                f"| {row['config']} | {row['setup']} | {row['class_linear_accuracy']:.4f} | "
                f"{row['raw_reconstruction_r2']:.3f} | {row['cka_to_raw_input']:.3f} | "
                f"{row['cka_to_labels']:.3f} | {row['view_retrieval_top1']:.3f} | "
                f"{row['class_knn_purity']:.3f} | {row['effective_rank']:.1f} | "
                f"{row['same_over_shuffled_mse']:.3f} |"
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
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_layer_rows = []
    all_setup_rows = []
    all_summary_rows = []
    all_content_rows = []

    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="barlow_clean",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=args.num_classes,
            )
            tensors = load_tensors(point, device)
            states = []
            if args.saved_model is not None and args.saved_model.exists():
                saved = load_saved_barlow(args.saved_model)
                saved["config"]["name"] = f"saved_{saved['config']['name']}"
                states.append(("bt_saved", saved))
            for config_text in args.configs:
                config = parse_config(config_text)
                print(f"training BT seed={seed} depth={depth} dataset={args.dataset} config={config}", flush=True)
                state = train_barlow(point, tensors, device, config, args.print_every_epochs)
                model_name = f"bt_{config['name']}_d{depth}_{config_hash(config)}"
                torch.save(
                    {**state, "point": asdict(point), "model_type": "backprop_barlow_twins_tuned"},
                    args.out_dir / f"{model_name}.pt",
                )
                states.append((model_name, state))
                torch.cuda.empty_cache()

            for model_name, state in states:
                config = state["config"]
                reps = collect_barlow_representations(point, tensors, state, device)
                layer_rows, setup_rows = readout_rows(
                    point,
                    model_name,
                    config,
                    reps,
                    tensors["ytr_np"],
                    tensors["yte_np"],
                    args.probe_reg,
                    args.pca_dim,
                    include_pca=not args.layer_only,
                )
                content = []
                if not args.layer_only:
                    content = content_rows(
                        point,
                        model_name,
                        config,
                        reps,
                        tensors["xtr_np"],
                        tensors["xte_np"],
                        tensors["ytr_np"],
                        tensors["yte_np"],
                        args.probe_reg,
                        args.recon_reg,
                        args.pca_dim,
                        args.max_retrieval,
                        args.knn_k,
                    )
                all_layer_rows.extend(layer_rows)
                all_setup_rows.extend(setup_rows)
                all_summary_rows.append(summarize_setup(point, model_name, config, layer_rows + setup_rows))
                all_content_rows.extend(content)
                write_jsonl(args.out_dir / "layer_readouts.partial.jsonl", all_layer_rows)
                write_jsonl(args.out_dir / "setup_readouts.partial.jsonl", all_setup_rows)
                write_jsonl(args.out_dir / "summary.partial.jsonl", all_summary_rows)
                write_jsonl(args.out_dir / "content_rows.partial.jsonl", all_content_rows)
                del reps, layer_rows, setup_rows, content
                gc.collect()
                torch.cuda.empty_cache()

    write_jsonl(args.out_dir / "layer_readouts.jsonl", all_layer_rows)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", all_setup_rows)
    write_jsonl(args.out_dir / "summary.jsonl", all_summary_rows)
    write_jsonl(args.out_dir / "content_rows.jsonl", all_content_rows)
    print(build_report(args.out_dir, all_summary_rows, all_content_rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Clean Barlow Twins readouts with same-instance CIFAR100 positives.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_barlow_clean"))
    parser.add_argument("--saved-model", type=Path, default=Path("docs/cf_mlp_representation_learning/models_resized_seed7/04_barlow_twins.pt"))
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
    parser.add_argument("--recon-reg", type=float, default=100.0)
    parser.add_argument("--max-retrieval", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--print-every-epochs", type=float, default=10.0)
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "proj2048_l0.005:100:0.001:1024:2048:0.005:0.0001",
            "proj2048_l0.001:100:0.001:1024:2048:0.001:0.0001",
        ],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
