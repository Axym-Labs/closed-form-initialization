import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from cf_mlp_clean_readouts import collect_depth_representations, linear_classifier_readout
from cf_mlp_last_layer_content import (
    class_knn_metrics,
    class_one_hot,
    linear_cka,
    raw_reconstruction_metrics,
    view_retrieval_metrics,
)
from cf_mlp_layer_mechanistic import covariance_spectrum, view_alignment
from cf_mlp_scalability import SweepPoint, write_jsonl


def setup_content_rows(state, variant, probe_reg, recon_reg, pca_dim, max_retrieval, knn_k):
    ytr = state["ytr"]
    yte = state["yte"]
    xtr = state["xtr"]
    xte = state["xte"]
    point = state["point"]
    labels_onehot = class_one_hot(ytr, int(point["num_classes"]))
    first = state["pathnorm_train"][0]
    features = [
        (
            "last_layer_512" if state["pathnorm_train"][-1].shape[1] == pca_dim else "last_layer_pca512",
            "cf_last_hidden_512" if state["pathnorm_train"][-1].shape[1] == pca_dim else "cf_last_hidden_to_pca512",
            state["pathnorm_train"][-1],
            state["pathnorm_test"][-1],
            state["pathnorm_view1_train"][-1],
            state["pathnorm_view2_train"][-1],
            1.0,
        )
    ]
    all_tr = np.concatenate(state["pathnorm_train"], axis=1)
    all_te = np.concatenate(state["pathnorm_test"], axis=1)
    all_v1 = np.concatenate(state["pathnorm_view1_train"], axis=1)
    all_v2 = np.concatenate(state["pathnorm_view2_train"], axis=1)
    pca = PCA(n_components=min(pca_dim, all_tr.shape[1]), svd_solver="randomized", iterated_power=3, random_state=int(point["seed"]) + 313)
    all_tr_pca = pca.fit_transform(all_tr).astype(np.float32)
    all_te_pca = pca.transform(all_te).astype(np.float32)
    all_v1_pca = pca.transform(all_v1).astype(np.float32)
    all_v2_pca = pca.transform(all_v2).astype(np.float32)
    features.append(
        (
            "all_layers_pca512",
            "cf_all_hidden_concat_to_pca512",
            all_tr_pca,
            all_te_pca,
            all_v1_pca,
            all_v2_pca,
            float(np.sum(pca.explained_variance_ratio_)),
        )
    )
    rows = []
    for setup, representation, ftr, fte, v1, v2, explained in features:
        readout = linear_classifier_readout(ftr, fte, ytr, yte, probe_reg)
        row = {
            "variant": variant,
            "seed": int(point["seed"]),
            "input_dim": int(point["input_dim"]),
            "width": int(point["width"]),
            "depth": int(point["depth"]),
            "transform_kind": state["transform_kind"],
            "schedule_name": state["schedule_name"],
            "activation": state["activation"],
            "activation_alpha": float(state["activation_alpha"]),
            "setup": setup,
            "representation": representation,
            "supervised_mapping": "single_linear_classifier",
            "class_linear_accuracy": readout["test_accuracy"],
            "class_linear_train_accuracy": readout["train_accuracy"],
            "pca_explained_variance": explained,
        }
        row.update(covariance_spectrum(ftr))
        row.update(view_alignment(v1, v2, int(point["seed"]) + 6262))
        row.update(raw_reconstruction_metrics(ftr, fte, xtr, xte, recon_reg))
        row.update(view_retrieval_metrics(v1, v2, max_retrieval))
        row.update(class_knn_metrics(fte, yte, max_retrieval, knn_k))
        row["cka_to_raw_input"] = linear_cka(ftr, xtr)
        row["cka_to_labels"] = linear_cka(ftr, labels_onehot)
        row["cka_to_first_layer"] = linear_cka(ftr, first)
        rows.append(row)
    return rows


def build_report(out_dir, rows):
    lines = [
        "# CF Setup Representation Content",
        "",
        "Rows compare the two clean CF representation setups. Each supervised mapping is one linear classifier on the frozen representation.",
        "",
        "| Variant | Depth | Setup | Class acc | Raw recon R2 | CKA raw | CKA labels | View top1 | kNN class | Rank | View ratio |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['depth']} | {row['setup']} | {row['class_linear_accuracy']:.4f} | "
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
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in args.depths:
        for seed in args.seeds:
            point = SweepPoint(
                dataset="cifar100_fullres_width",
                axis="cf_setup_content",
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
                print(f"CF setup content seed={seed} depth={depth} variant={variant}", flush=True)
                state = collect_depth_representations(
                    point,
                    device,
                    device_name,
                    transform_kind=transform_kind,
                    schedule_name=schedule_name,
                    activation_name=activation_name,
                )
                rows.extend(
                    setup_content_rows(
                        state,
                        variant,
                        args.probe_reg,
                        args.recon_reg,
                        args.pca_dim,
                        args.max_retrieval,
                        args.knn_k,
                    )
                )
                write_jsonl(args.out_dir / "content_rows.partial.jsonl", rows)
                del state
                gc.collect()
                torch.cuda.empty_cache()
    write_jsonl(args.out_dir / "content_rows.jsonl", rows)
    print(build_report(args.out_dir, rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Content metrics for clean CF last-layer and all-layer PCA setups.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/cf_mlp_representation_learning/artifacts_cf_setup_content"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=0.40)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depths", type=int, nargs="+", default=[6])
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--probe-reg", type=float, default=100.0)
    parser.add_argument("--recon-reg", type=float, default=100.0)
    parser.add_argument("--max-retrieval", type=int, default=2000)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=["cf:relax4:leaky0.2", "cf:relax4:leakygelu0.5"])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
