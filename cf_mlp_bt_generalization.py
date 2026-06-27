import argparse
import csv
import gc
from pathlib import Path

import torch

from cf_mlp_bt_objective_by_layer import bt_hidden_metrics, residual_cf_args
from cf_mlp_residual_bt_variants import collect_variant_state
from cf_mlp_scalability import SweepPoint, write_jsonl


def point_for(args, depth, seed):
    return SweepPoint(
        dataset=args.dataset,
        axis="bt_generalization",
        scale_value=args.width,
        seed=seed,
        n_train=args.n_train,
        n_test=args.n_test,
        input_dim=args.input_dim,
        width=args.width,
        depth=depth,
        num_classes=args.num_classes,
    )


def rows_for_state(state, args):
    rows = []
    split_pairs = {
        "train": (state["pathnorm_view1_train"], state["pathnorm_view2_train"]),
        "test": (state["pathnorm_view1_test"], state["pathnorm_view2_test"]),
    }
    for split, (view1_layers, view2_layers) in split_pairs.items():
        for idx, (view1, view2) in enumerate(zip(view1_layers, view2_layers)):
            row = {
                "variant": state["variant"],
                "kind": state["kind"],
                "activation": state["activation"],
                "activation_alpha": state["activation_alpha"],
                "dataset": state["point"]["dataset"],
                "seed": int(state["point"]["seed"]),
                "depth": int(state["point"]["depth"]),
                "width": int(state["point"]["width"]),
                "layer": idx + 1,
                "split": split,
            }
            row.update(bt_hidden_metrics(view1, view2, args.bt_lambda))
            rows.append(row)
    return rows


def write_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(args, rows, out_dir):
    lines = [
        "# CF-BT Train/Test Generalization",
        "",
        "Layerwise BT metrics on fitted train augmentation pairs and held-out test augmentation pairs.",
        f"BT lambda: `{args.bt_lambda}`.",
        f"TF32 enabled: `{not args.no_tf32}`.",
        "",
        "| Variant | Depth | Split | Final BT/dim | Best BT/dim | Best layer | Final corr diag | Final on/dim | Final weighted off/dim |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in args.variants:
        for depth in args.depths:
            for split in ("train", "test"):
                subset = [
                    row
                    for row in rows
                    if row["variant"] == variant and row["depth"] == depth and row["split"] == split
                ]
                if not subset:
                    continue
                subset = sorted(subset, key=lambda row: row["layer"])
                final = subset[-1]
                best = min(subset, key=lambda row: row["bt_total_per_dim"])
                lines.append(
                    f"| {variant} | {depth} | {split} | {final['bt_total_per_dim']:.4g} | "
                    f"{best['bt_total_per_dim']:.4g} | {best['layer']} | "
                    f"{final['corr_diag_mean']:.3f} | {final['bt_on_diag_per_dim']:.4g} | "
                    f"{final['bt_weighted_off_diag_per_dim']:.4g} |"
                )
    lines.extend(
        [
            "",
            "Files: `bt_generalization.jsonl`, `bt_generalization.csv`.",
            "",
        ]
    )
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return report


def run(args):
    torch.set_num_threads(args.torch_threads)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = point_for(args, depth, seed)
            for variant in args.variants:
                print(f"bt-generalization depth={depth} seed={seed} variant={variant}", flush=True)
                state = collect_variant_state(point, variant, residual_cf_args(args), device, device_name)
                all_rows.extend(rows_for_state(state, args))
                write_jsonl(args.out_dir / "bt_generalization.partial.jsonl", all_rows)
                del state
                torch.cuda.empty_cache()
                gc.collect()

    write_jsonl(args.out_dir / "bt_generalization.jsonl", all_rows)
    write_csv(args.out_dir / "bt_generalization.csv", all_rows)
    print(build_report(args, all_rows, args.out_dir), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train/test generalization for CF-BT paired-view objectives.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_bt_generalization_seed7"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset", default="cifar100_simclr")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--bt-lambda", type=float, default=0.005)
    parser.add_argument("--cf-residual-scale", type=float, default=1.0)
    parser.add_argument("--postrelu-fit-samples", type=int, default=1024)
    parser.add_argument("--postrelu-steps", type=int, default=40)
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
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["plain_cf_relu", "plain_cf_agreement_expand_fullwhiten_relu_k192"],
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
