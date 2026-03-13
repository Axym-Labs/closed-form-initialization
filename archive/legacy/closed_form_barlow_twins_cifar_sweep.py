import argparse
import json
from pathlib import Path

from closed_form_barlow_twins_cifar import run_experiment
from project_paths import default_json_path, resolve_json_path


DEFAULT_LAMBDAS = [0.1, 0.3, 1.0, 3.0]


def lambda_tag(value):
    return f"{value:.3f}".replace(".", "p")


def main():
    parser = argparse.ArgumentParser(description="Lambda sweep for CIFAR closed-form Barlow Twins.")
    parser.add_argument("--suite", choices=["single-translation", "block-masking"], default="single-translation")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--final-d", type=int, default=32)
    parser.add_argument("--n-train", type=int, default=6000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=default_json_path("closed_form_barlow_twins_cifar_sweep.json"),
    )
    args = parser.parse_args()

    cifar10_runs = []
    for lambda_reg in args.lambdas:
        result = run_experiment(
            dataset_name="cifar10",
            suite_name=args.suite,
            lambda_reg=lambda_reg,
            depth=args.depth,
            final_dim=args.final_d,
            n_train=args.n_train,
            n_test=args.n_test,
        )
        cifar10_runs.append(result)
        run_path = default_json_path(
            f"closed_form_barlow_twins_cifar10_{args.suite}_lambda_{lambda_tag(lambda_reg)}.json"
        )
        run_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    cifar10_runs.sort(key=lambda row: row["full_probe_accuracy"], reverse=True)
    best_lambda = cifar10_runs[0]["lambda"]

    cifar100_best = run_experiment(
        dataset_name="cifar100",
        suite_name=args.suite,
        lambda_reg=best_lambda,
        depth=args.depth,
        final_dim=args.final_d,
        n_train=args.n_train,
        n_test=args.n_test,
    )
    cifar100_path = default_json_path(
        f"closed_form_barlow_twins_cifar100_{args.suite}_lambda_{lambda_tag(best_lambda)}.json"
    )
    cifar100_path.write_text(json.dumps(cifar100_best, indent=2), encoding="utf-8")

    payload = {
        "config": {
            "suite": args.suite,
            "depth": args.depth,
            "final_d": args.final_d,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "lambdas": args.lambdas,
            "selection_metric": "full_probe_accuracy",
        },
        "cifar10_runs": cifar10_runs,
        "best_lambda": best_lambda,
        "cifar100_best_lambda_run": cifar100_best,
    }

    output_path = resolve_json_path(args.save_json)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"CIFAR-10 lambda sweep complete for suite={args.suite}")
    for result in cifar10_runs:
        print(
            f"lambda={result['lambda']:.3f} | full={result['full_probe_accuracy']:.4f} | "
            f"final_pca={result['final_pca_probe_accuracy']:.4f} | raw_pca={result['raw_input_pca_probe_accuracy']:.4f}"
        )
    print(f"best lambda by full probe: {best_lambda:.3f}")
    print(
        f"CIFAR-100 | lambda={cifar100_best['lambda']:.3f} | full={cifar100_best['full_probe_accuracy']:.4f} | "
        f"final_pca={cifar100_best['final_pca_probe_accuracy']:.4f} | raw_pca={cifar100_best['raw_input_pca_probe_accuracy']:.4f}"
    )
    print(f"saved summary to {output_path}")


if __name__ == "__main__":
    main()
