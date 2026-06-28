import argparse
import csv
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cf_mlp_barlow_clean import load_tensors, normalized_inputs
from cf_mlp_bt_objective_by_layer import find_residual_bt_model
from cf_mlp_moment_ols_residual import bt_corr_and_gradient
from cf_mlp_residual_barlow import leaky_gelu
from cf_mlp_scalability import SweepPoint, write_jsonl


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="bp_moment_velocity",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def residual_layer(h, weight, activation_alpha, residual_scale, use_layernorm):
    branch = leaky_gelu(h @ weight, activation_alpha)
    out = h + float(residual_scale) * branch
    if use_layernorm:
        out = F.layer_norm(out, (out.shape[-1],))
    return out


def offdiag(matrix):
    return matrix - torch.diag(torch.diagonal(matrix))


def cosine(a, b):
    af = a.reshape(-1)
    bf = b.reshape(-1)
    return float((torch.dot(af, bf) / torch.clamp(torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf), min=1e-12)).detach().cpu().item())


def bt_loss_from_corr(corr, bt_lambda):
    diag = torch.diagonal(corr)
    on = torch.sum((diag - 1.0) ** 2)
    off = torch.sum(corr * corr) - torch.sum(diag * diag)
    return (on + float(bt_lambda) * off) / corr.shape[0]


def offdiag_rms(matrix):
    off = offdiag(matrix)
    denom = max(1, matrix.numel() - matrix.shape[0])
    return float(torch.sqrt(torch.sum(off * off) / denom).detach().cpu().item())


def effective_rank_from_svals(svals):
    total = torch.sum(torch.clamp(svals, min=0.0))
    if float(total.detach().cpu().item()) <= 1e-12:
        return 0.0
    probs = svals / torch.clamp(total, min=1e-12)
    entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-12)))
    return float(torch.exp(entropy).detach().cpu().item())


def polar_target(corr):
    u, _, vh = torch.linalg.svd(corr, full_matrices=False)
    return u @ vh - corr


def identity_target(corr):
    eye = torch.eye(corr.shape[0], dtype=corr.dtype, device=corr.device)
    return eye - corr


def diag_identity_target(corr):
    out = torch.zeros_like(corr)
    diag = torch.arange(corr.shape[0], device=corr.device)
    out[diag, diag] = 1.0 - corr[diag, diag]
    return out


def row_for_layer(args, point, model, layer_idx, before1, before2, after1, after2):
    _, _, corr, grad, _, _ = bt_corr_and_gradient(before1, before2, args.bt_lambda)
    _, _, corr_after, _, _, _ = bt_corr_and_gradient(after1, after2, args.bt_lambda)
    delta = corr_after - corr
    neg_grad = -grad
    ident = identity_target(corr)
    diag_ident = diag_identity_target(corr)
    polar = polar_target(corr)
    before_loss = bt_loss_from_corr(corr, args.bt_lambda)
    after_loss = bt_loss_from_corr(corr_after, args.bt_lambda)
    first_order = torch.sum(grad * delta) / corr.shape[0]
    delta_diag = torch.diagonal(delta)
    svals = torch.linalg.svdvals(corr_after)
    return {
        "model": model,
        "dataset": point.dataset,
        "seed": point.seed,
        "depth": point.depth,
        "width": point.width,
        "layer": layer_idx + 1,
        "bt_before_per_dim": float(before_loss.detach().cpu().item()),
        "bt_after_per_dim": float(after_loss.detach().cpu().item()),
        "bt_delta_per_dim": float((after_loss - before_loss).detach().cpu().item()),
        "bt_first_order_delta_per_dim": float(first_order.detach().cpu().item()),
        "corr_diag_before_mean": float(torch.diagonal(corr).mean().detach().cpu().item()),
        "corr_diag_after_mean": float(torch.diagonal(corr_after).mean().detach().cpu().item()),
        "corr_diag_delta_mean": float(delta_diag.mean().detach().cpu().item()),
        "corr_offdiag_before_rms": offdiag_rms(corr),
        "corr_offdiag_after_rms": offdiag_rms(corr_after),
        "corr_delta_norm": float(torch.linalg.vector_norm(delta).detach().cpu().item()),
        "corr_delta_diag_norm_frac": float(
            (torch.linalg.vector_norm(torch.diag(delta_diag)) / torch.clamp(torch.linalg.vector_norm(delta), min=1e-12))
            .detach()
            .cpu()
            .item()
        ),
        "delta_cos_neg_bt_grad": cosine(delta, neg_grad),
        "delta_cos_identity_minus_c": cosine(delta, ident),
        "delta_cos_diag_identity": cosine(delta, diag_ident),
        "delta_cos_polar_minus_c": cosine(delta, polar),
        "neg_bt_grad_cos_identity_minus_c": cosine(neg_grad, ident),
        "neg_bt_grad_cos_polar_minus_c": cosine(neg_grad, polar),
        "identity_cos_polar_minus_c": cosine(ident, polar),
        "corr_after_singular_mean": float(torch.mean(svals).detach().cpu().item()),
        "corr_after_singular_max": float(torch.max(svals).detach().cpu().item()),
        "corr_after_singular_effective_rank": effective_rank_from_svals(svals),
    }


def find_greedy_model(model_dir, depth):
    matches = sorted(model_dir.glob(f"*d{depth}.pt"))
    if not matches:
        raise FileNotFoundError(f"No greedy residual BP-BT checkpoint for depth {depth} in {model_dir}")
    return matches[-1]


def load_model_state(args, model, depth):
    if model == "e2e_residual_bpbt":
        path = find_residual_bt_model(args.e2e_model_dir, depth)
    elif model == "greedy_residual_bpbt":
        path = find_greedy_model(args.greedy_model_dir, depth)
    else:
        raise ValueError(f"Unknown model: {model}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    return path, state


def rows_for_model(args, point, tensors, model, device):
    path, state = load_model_state(args, model, point.depth)
    norm_mean = state["norm_mean"].to(device)
    norm_scale = state["norm_scale"].to(device)
    normed = normalized_inputs(tensors, norm_mean, norm_scale)
    view1 = normed["view1_tr"]
    view2 = normed["view2_tr"]
    weights = [param.to(device) for param in state["weights"]]
    rows = []
    print(f"moment velocity model={model} depth={point.depth} checkpoint={path}", flush=True)
    with torch.no_grad():
        for layer_idx, weight in enumerate(weights):
            before1 = view1
            before2 = view2
            after1 = residual_layer(
                before1,
                weight,
                state["activation_alpha"],
                state["residual_scale"],
                state["layernorm"],
            )
            after2 = residual_layer(
                before2,
                weight,
                state["activation_alpha"],
                state["residual_scale"],
                state["layernorm"],
            )
            row = row_for_layer(args, point, model, layer_idx, before1, before2, after1, after2)
            row["checkpoint"] = str(path)
            row["activation_alpha"] = float(state["activation_alpha"])
            row["residual_scale"] = float(state["residual_scale"])
            row["layernorm"] = bool(state["layernorm"])
            rows.append(row)
            view1 = after1
            view2 = after2
    return rows


def summarize(rows):
    summaries = []
    keys = sorted({(row["model"], row["depth"], row["seed"]) for row in rows})
    for model, depth, seed in keys:
        items = sorted(
            [row for row in rows if row["model"] == model and row["depth"] == depth and row["seed"] == seed],
            key=lambda row: row["layer"],
        )
        first = items[0]
        final = items[-1]
        summaries.append(
            {
                "model": model,
                "depth": depth,
                "seed": seed,
                "first_bt_per_dim": first["bt_before_per_dim"],
                "final_bt_per_dim": final["bt_after_per_dim"],
                "bt_improving_step_fraction": float(np.mean([row["bt_delta_per_dim"] < 0.0 for row in items])),
                "mean_bt_delta_per_dim": float(np.mean([row["bt_delta_per_dim"] for row in items])),
                "mean_first_order_delta_per_dim": float(
                    np.mean([row["bt_first_order_delta_per_dim"] for row in items])
                ),
                "first_corr_diag": first["corr_diag_before_mean"],
                "final_corr_diag": final["corr_diag_after_mean"],
                "first_offdiag_rms": first["corr_offdiag_before_rms"],
                "final_offdiag_rms": final["corr_offdiag_after_rms"],
                "mean_delta_cos_neg_bt_grad": float(np.mean([row["delta_cos_neg_bt_grad"] for row in items])),
                "mean_delta_cos_identity_minus_c": float(
                    np.mean([row["delta_cos_identity_minus_c"] for row in items])
                ),
                "mean_delta_cos_diag_identity": float(np.mean([row["delta_cos_diag_identity"] for row in items])),
                "mean_delta_cos_polar_minus_c": float(np.mean([row["delta_cos_polar_minus_c"] for row in items])),
                "mean_corr_delta_diag_norm_frac": float(
                    np.mean([row["corr_delta_diag_norm_frac"] for row in items])
                ),
                "final_corr_singular_effective_rank": final["corr_after_singular_effective_rank"],
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
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def write_report(path, summaries):
    lines = [
        "# BP-BT Moment Velocity Law",
        "",
        "For each residual BP-BT layer, this measures the realized cross-correlation velocity "
        "`Delta C = C_{l+1} - C_l` and compares it to simple moment-space laws.",
        "",
        "| Model | Depth | BT first->final | Improve frac | Corr diag first->final | Offdiag RMS first->final | cos dC,-grad | cos dC,I-C | cos dC,diag(I-C) | cos dC,polar-C | diag frac | Corr-rank |",
        "| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(summaries, key=lambda item: (item["depth"], item["model"])):
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["depth"]),
                    f"{fmt(row['first_bt_per_dim'])}->{fmt(row['final_bt_per_dim'])}",
                    fmt(row["bt_improving_step_fraction"]),
                    f"{fmt(row['first_corr_diag'])}->{fmt(row['final_corr_diag'])}",
                    f"{fmt(row['first_offdiag_rms'])}->{fmt(row['final_offdiag_rms'])}",
                    fmt(row["mean_delta_cos_neg_bt_grad"]),
                    fmt(row["mean_delta_cos_identity_minus_c"]),
                    fmt(row["mean_delta_cos_diag_identity"]),
                    fmt(row["mean_delta_cos_polar_minus_c"]),
                    fmt(row["mean_corr_delta_diag_norm_frac"]),
                    fmt(row["final_corr_singular_effective_rank"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: a viable CF target should match BP's `Delta C` direction, not merely reduce BT in isolation.",
            "Files: `moment_velocity_rows.jsonl/csv`, `summary.jsonl/csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = point_for(args, depth, seed)
            tensors = load_tensors(point, device)
            for model in args.models:
                all_rows.extend(rows_for_model(args, point, tensors, model, device))
                write_jsonl(args.out_dir / "moment_velocity_rows.partial.jsonl", all_rows)
                torch.cuda.empty_cache()
                gc.collect()
            del tensors
            torch.cuda.empty_cache()
            gc.collect()
    summaries = summarize(all_rows)
    write_jsonl(args.out_dir / "moment_velocity_rows.jsonl", all_rows)
    write_jsonl(args.out_dir / "summary.jsonl", summaries)
    write_csv(args.out_dir / "moment_velocity_rows.csv", all_rows)
    write_csv(args.out_dir / "summary.csv", summaries)
    write_report(args.out_dir / "report.md", summaries)
    print((args.out_dir / "report.md").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Measure residual BP-BT's layerwise moment velocity law.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_bp_moment_velocity_seed7"))
    parser.add_argument("--e2e-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_residual_barlow_seed7"))
    parser.add_argument("--greedy-model-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_route3_residual_bt_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=["e2e_residual_bpbt", "greedy_residual_bpbt"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
