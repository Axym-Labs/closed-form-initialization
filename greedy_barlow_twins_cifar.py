import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import closed_form_barlow_twins as cfbt
import cifar_shared
from experiment_settings import (
    ACTIVATION,
    BATCH_SIZE,
    DATASETS,
    DEPTH,
    GREEDY_BT_EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    N_TEST,
    N_TRAIN,
    SEED,
    SUITES,
    W,
    WEIGHT_DECAY,
)
from project_paths import resolve_json_path


WIDTHS = [W] * DEPTH
LR = LEARNING_RATE
LAMBDA_OFFDIAG = 5e-3
PROBE_MAX_ITER = 2000
LEAKY_RELU_SLOPE = 0.1


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_activation_torch(x, activation):
    if activation == "relu":
        return torch.relu(x)
    if activation == "tanh":
        return torch.tanh(x)
    if activation == "leaky-relu":
        return torch.where(x >= 0.0, x, LEAKY_RELU_SLOPE * x)
    if activation == "identity":
        return x
    raise ValueError(f"Unknown activation: {activation}")


def off_diagonal(x):
    n, m = x.shape
    if n != m:
        raise ValueError("Expected square correlation matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1, z2, lambda_offdiag):
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)
    c = z1.T @ z2 / z1.shape[0]
    on_diag = torch.diagonal(c).add(-1.0).pow(2).sum()
    off = off_diagonal(c).pow(2).sum()
    return on_diag + lambda_offdiag * off


class GreedyBarlowMLP(nn.Module):
    def __init__(self, input_dim, widths, activation):
        super().__init__()
        dims = [input_dim] + list(widths)
        self.activation = activation
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(widths))]
        )

    def forward_to_layer_input(self, x, layer_idx):
        h = x
        for idx in range(layer_idx):
            h = apply_activation_torch(self.layers[idx](h), self.activation)
        return h

    def extract_features(self, x):
        h = x
        for layer in self.layers:
            h = apply_activation_torch(layer(h), self.activation)
        return h

    def extract_features_upto(self, x, depth):
        h = x
        for layer_idx in range(depth):
            h = apply_activation_torch(self.layers[layer_idx](h), self.activation)
        return h


def train_layer_greedy(model, layer_idx, loader, device, loss_position, epochs, lr, momentum, weight_decay, lambda_offdiag):
    for idx, layer in enumerate(model.layers):
        for param in layer.parameters():
            param.requires_grad = idx == layer_idx

    optimizer = torch.optim.SGD(
        model.layers[layer_idx].parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    model.train()
    epoch_losses = []
    for _ in range(epochs):
        losses = []
        for x1, x2 in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            with torch.no_grad():
                h1 = model.forward_to_layer_input(x1, layer_idx)
                h2 = model.forward_to_layer_input(x2, layer_idx)

            pre1 = model.layers[layer_idx](h1)
            pre2 = model.layers[layer_idx](h2)
            if loss_position == "pre":
                z1, z2 = pre1, pre2
            else:
                z1 = apply_activation_torch(pre1, model.activation)
                z2 = apply_activation_torch(pre2, model.activation)

            loss = barlow_twins_loss(z1, z2, lambda_offdiag=lambda_offdiag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
    return epoch_losses


def fit_linear_probe(ztr, ytr, zte, yte):
    kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
        "n_jobs": None,
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        kwargs["multi_class"] = "multinomial"
    clf = LogisticRegression(**kwargs)
    clf.fit(ztr, ytr)
    return float((clf.predict(zte) == yte).mean())


def run_variant(
    dataset_name,
    suite_name,
    loss_position,
    widths,
    batch_size,
    epochs,
    lr,
    momentum,
    weight_decay,
    lambda_offdiag,
    activation,
    n_train,
    n_test,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = cifar_shared.load_cifar_numpy(dataset_name, n_train=n_train, n_test=n_test, seed=SEED, width=widths[0])
    xtr_img, ytr, xte_img, yte = dataset["xtr_img"], dataset["ytr"], dataset["xte_img"], dataset["yte"]
    xtr = dataset["xtr"]
    xte = dataset["xte"]
    x1tr, x2tr = cifar_shared.sample_pair_views(xtr_img, suite_name, seed=SEED + 101, width=widths[0], repeats=1, mean=dataset["mean"])

    train_ds = TensorDataset(torch.from_numpy(x1tr).float(), torch.from_numpy(x2tr).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = GreedyBarlowMLP(xtr.shape[1], widths, activation=activation).to(device)
    layer_stats = []
    for layer_idx in range(len(widths)):
        losses = train_layer_greedy(
            model,
            layer_idx,
            train_loader,
            device=device,
            loss_position=loss_position,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            lambda_offdiag=lambda_offdiag,
        )
        model.eval()
        with torch.no_grad():
            ztr = model.extract_features_upto(torch.from_numpy(xtr).float().to(device), layer_idx + 1).cpu().numpy()
            zte = model.extract_features_upto(torch.from_numpy(xte).float().to(device), layer_idx + 1).cpu().numpy()
        ztr_std, zte_std = cfbt.standardize_train_test(ztr, zte)
        probe_acc = fit_linear_probe(ztr_std, ytr, zte_std, yte)
        layer_stats.append(
            {
                "depth": layer_idx + 1,
                "probe_accuracy": probe_acc,
                "final_epoch_loss": losses[-1],
                "loss_curve": losses,
            }
        )

    probe_acc = layer_stats[-1]["probe_accuracy"]
    return {
        "dataset": dataset_name,
        "suite": suite_name,
        "loss_position": loss_position,
        "activation": activation,
        "widths": widths,
        "n_train": n_train,
        "n_test": n_test,
        "probe_accuracy": probe_acc,
        "depth_metrics": layer_stats,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="Greedy layerwise Barlow Twins MLP on CIFAR.")
    parser.add_argument("--dataset", choices=DATASETS, default=DATASETS[0])
    parser.add_argument("--suite", default=SUITES[0], choices=SUITES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=GREEDY_BT_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-offdiag", type=float, default=LAMBDA_OFFDIAG)
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky-relu", "identity"], default=ACTIVATION)
    parser.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    parser.add_argument("--loss-position", choices=["post"], default="post")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    result = run_variant(
        dataset_name=args.dataset,
        suite_name=args.suite,
        loss_position="post",
        widths=args.widths,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lambda_offdiag=args.lambda_offdiag,
        activation=args.activation,
        n_train=args.n_train,
        n_test=args.n_test,
    )

    print(
        f"Greedy Barlow Twins CIFAR  |  dataset={args.dataset}  |  suite={args.suite}  |  activation={args.activation}  |  widths={args.widths}"
    )
    print(f"{result['loss_position']:>4s} activation loss | probe accuracy = {result['probe_accuracy']:.4f}")
    for layer in result["depth_metrics"]:
        print(
            f"  depth {layer['depth']} probe = {layer['probe_accuracy']:.4f} | final loss = {layer['final_epoch_loss']:.4f}"
        )

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
