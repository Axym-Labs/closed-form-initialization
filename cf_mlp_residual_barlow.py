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

from cf_mlp_barlow_clean import barlow_loss, load_tensors, normalized_inputs, parse_config
from cf_mlp_clean_readouts import linear_classifier_readout, pca512_readout
from cf_mlp_layer_mechanistic import covariance_spectrum, standardize_many, view_alignment
from cf_mlp_scalability import SweepPoint, write_jsonl
from cf_mlp_scalability_gpu import init_backprop_params


BP_BT_ACTIVATION = "leaky_gelu"
BP_BT_ACTIVATION_ALPHA = 0.5


def config_hash(config):
    text = json.dumps(config, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def leaky_gelu(x, alpha):
    return F.gelu(x) + float(alpha) * torch.minimum(x, torch.zeros((), dtype=x.dtype, device=x.device))


def forward_residual_hiddens(x, weights, activation_alpha, residual_scale, use_layernorm):
    h = x
    hiddens = []
    for weight in weights:
        branch = leaky_gelu(h @ weight, activation_alpha)
        h = h + float(residual_scale) * branch
        if use_layernorm:
            h = F.layer_norm(h, (h.shape[-1],))
        hiddens.append(h)
    return hiddens


def train_residual_barlow(point, tensors, device, config, args):
    tuned_point = replace(point, lr=config["lr"], weight_decay=config["weight_decay"])
    normed = normalized_inputs(tensors)
    weights, _ = init_backprop_params(tuned_point, device)
    if args.branch_init_scale != 1.0:
        with torch.no_grad():
            for weight in weights:
                weight.mul_(float(args.branch_init_scale))
    final_dim = min(point.width, point.input_dim)
    torch.manual_seed(point.seed + 1707)
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
        h1 = forward_residual_hiddens(
            normed["view1_tr"][batch_idx],
            weights,
            args.activation_alpha,
            args.residual_scale,
            args.layernorm,
        )[-1]
        h2 = forward_residual_hiddens(
            normed["view2_tr"][batch_idx],
            weights,
            args.activation_alpha,
            args.residual_scale,
            args.layernorm,
        )[-1]
        loss = barlow_loss(h1 @ projector, h2 @ projector, config["bt_lambda"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if step % steps_per_epoch == 0 or step == total_steps:
            epoch = step / steps_per_epoch
            recent = float(np.mean(losses[-min(len(losses), steps_per_epoch) :]))
            if epoch % args.print_every_epochs < 1e-9 or step == total_steps:
                print(f"residual-bt {config['name']} depth={point.depth} epoch={epoch:.1f} loss={recent:.3f}", flush=True)
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
        "activation": BP_BT_ACTIVATION,
        "activation_alpha": float(args.activation_alpha),
        "residual_scale": float(args.residual_scale),
        "layernorm": bool(args.layernorm),
        "branch_init_scale": float(args.branch_init_scale),
    }


def collect_residual_representations(point, tensors, state, device):
    norm_mean = state["norm_mean"].to(device)
    norm_scale = state["norm_scale"].to(device)
    normed = normalized_inputs(tensors, norm_mean, norm_scale)
    weights = [param.to(device) for param in state["weights"]]
    with torch.no_grad():
        train_h = forward_residual_hiddens(
            normed["xtr"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        test_h = forward_residual_hiddens(
            normed["xte"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        view1_h = forward_residual_hiddens(
            normed["view1_tr"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
        view2_h = forward_residual_hiddens(
            normed["view2_tr"], weights, state["activation_alpha"], state["residual_scale"], state["layernorm"]
        )
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


def readout_rows(point, state, reps, ytr, yte, probe_reg, pca_dim):
    layer_rows = []
    for idx, (xtr, xte, v1, v2) in enumerate(
        zip(
            reps["pathnorm_train"],
            reps["pathnorm_test"],
            reps["pathnorm_view1_train"],
            reps["pathnorm_view2_train"],
        )
    ):
        row = {
            "model": "residual_backprop_barlow",
            "config": state["config"]["name"],
            "seed": point.seed,
            "dataset": point.dataset,
            "input_dim": point.input_dim,
            "width": point.width,
            "depth": point.depth,
            "layer": idx + 1,
            "setup": "layer_hidden_512",
            "activation": state["activation"],
            "activation_alpha": state["activation_alpha"],
            "residual_scale": state["residual_scale"],
            "layernorm": state["layernorm"],
        }
        row.update(linear_classifier_readout(xtr, xte, ytr, yte, probe_reg))
        row.update(covariance_spectrum(xtr))
        row.update(view_alignment(v1, v2, point.seed + idx + 9181))
        layer_rows.append(row)

    last_tr = reps["pathnorm_train"][-1]
    last_te = reps["pathnorm_test"][-1]
    setup_rows = []
    last = {
        "model": "residual_backprop_barlow",
        "config": state["config"]["name"],
        "seed": point.seed,
        "dataset": point.dataset,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "setup": "last_layer_512",
        "source_dim": int(last_tr.shape[1]),
    }
    last.update(linear_classifier_readout(last_tr, last_te, ytr, yte, probe_reg))
    setup_rows.append(last)

    all_tr = np.concatenate(reps["pathnorm_train"], axis=1)
    all_te = np.concatenate(reps["pathnorm_test"], axis=1)
    all_pca = {
        "model": "residual_backprop_barlow",
        "config": state["config"]["name"],
        "seed": point.seed,
        "dataset": point.dataset,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "setup": "all_layers_pca512",
    }
    all_pca.update(pca512_readout(all_tr, all_te, ytr, yte, probe_reg, point.seed, pca_dim))
    setup_rows.append(all_pca)

    best_layer = max(layer_rows, key=lambda row: row["test_accuracy"])
    last_layer = layer_rows[-1]
    summary = {
        "model": "residual_backprop_barlow",
        "config": state["config"]["name"],
        "seed": point.seed,
        "dataset": point.dataset,
        "input_dim": point.input_dim,
        "width": point.width,
        "depth": point.depth,
        "epochs": float(state["config"]["epochs"]),
        "lr": float(state["config"]["lr"]),
        "batch_size": int(state["config"]["batch_size"]),
        "projector_dim": int(state["config"]["projector_dim"]),
        "bt_lambda": float(state["config"]["bt_lambda"]),
        "weight_decay": float(state["config"]["weight_decay"]),
        "activation": state["activation"],
        "activation_alpha": float(state["activation_alpha"]),
        "residual_scale": float(state["residual_scale"]),
        "layernorm": bool(state["layernorm"]),
        "branch_init_scale": float(state["branch_init_scale"]),
        "fit_time_sec": float(state["fit_time_sec"]),
        "final_barlow_loss": float(state["history"][-1]["mean_barlow_loss"]) if state["history"] else float("nan"),
        "last_layer_accuracy": last["test_accuracy"],
        "all_pca_accuracy": all_pca["test_accuracy"],
        "all_pca_explained_variance": all_pca["explained_variance"],
        "best_layer_accuracy": best_layer["test_accuracy"],
        "best_layer": int(best_layer["layer"]),
        "last_minus_first_accuracy": last_layer["test_accuracy"] - layer_rows[0]["test_accuracy"],
        "last_layer_effective_rank": last_layer["effective_rank"],
        "last_layer_view_mse_ratio": last_layer["same_over_shuffled_mse"],
        "last_layer_view_cosine": last_layer["same_view_cosine"],
    }
    return layer_rows, setup_rows, summary


def build_report(out_dir, summaries):
    lines = [
        "# Residual Backprop Barlow Twins",
        "",
        "Architecture: `H <- LayerNorm(H + residual_scale * leaky_gelu(HW))`; readouts are frozen linear probes.",
        "",
        "| Depth | Last 512 | All PCA512 | Best layer | Best layer idx | Last-first | Last rank | View ratio | Final BT loss | Fit sec |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: (item["depth"], item["config"], item["seed"])):
        lines.append(
            f"| {row['depth']} | {row['last_layer_accuracy']:.4f} | {row['all_pca_accuracy']:.4f} | "
            f"{row['best_layer_accuracy']:.4f} | {row['best_layer']} | "
            f"{row['last_minus_first_accuracy']:+.4f} | {row['last_layer_effective_rank']:.1f} | "
            f"{row['last_layer_view_mse_ratio']:.3f} | {row['final_barlow_loss']:.3f} | {row['fit_time_sec']:.1f} |"
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
    config = parse_config(args.config)

    all_layer_rows = []
    all_setup_rows = []
    all_summaries = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset=args.dataset,
                axis="residual_barlow",
                scale_value=args.width,
                seed=seed,
                n_train=args.n_train,
                n_test=args.n_test,
                input_dim=args.input_dim,
                width=args.width,
                depth=depth,
                num_classes=args.num_classes,
            )
            print(
                f"training residual BT seed={seed} dataset={args.dataset} depth={depth} width={args.width} config={config}",
                flush=True,
            )
            tensors = load_tensors(point, device)
            state = train_residual_barlow(point, tensors, device, config, args)
            model_name = f"residual_bt_{config['name']}_d{depth}_{config_hash(config)}"
            torch.save({**state, "point": asdict(point), "model_type": "residual_backprop_barlow_twins"}, args.out_dir / f"{model_name}.pt")
            reps = collect_residual_representations(point, tensors, state, device)
            layer_rows, setup_rows, summary = readout_rows(
                point,
                state,
                reps,
                tensors["ytr_np"],
                tensors["yte_np"],
                args.probe_reg,
                args.pca_dim,
            )
            all_layer_rows.extend(layer_rows)
            all_setup_rows.extend(setup_rows)
            all_summaries.append(summary)
            write_jsonl(args.out_dir / "layer_readouts.partial.jsonl", all_layer_rows)
            write_jsonl(args.out_dir / "setup_readouts.partial.jsonl", all_setup_rows)
            write_jsonl(args.out_dir / "summary.partial.jsonl", all_summaries)
            del tensors, state, reps, layer_rows, setup_rows, summary
            torch.cuda.empty_cache()
            gc.collect()

    write_jsonl(args.out_dir / "layer_readouts.jsonl", all_layer_rows)
    write_jsonl(args.out_dir / "setup_readouts.jsonl", all_setup_rows)
    write_jsonl(args.out_dir / "summary.jsonl", all_summaries)
    print(build_report(args.out_dir, all_summaries), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Residual MLP Barlow Twins baseline with clean representation probes.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.38)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--config", default="residual_bt:100:0.001:1024:2048:0.005:0.0001")
    parser.add_argument("--activation-alpha", type=float, default=BP_BT_ACTIVATION_ALPHA)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--branch-init-scale", type=float, default=1.0)
    parser.add_argument("--layernorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--print-every-epochs", type=float, default=10.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
