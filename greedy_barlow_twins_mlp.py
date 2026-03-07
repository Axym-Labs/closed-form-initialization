import argparse
import json
from inspect import signature
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import test2


SEED = 7
N_TRAIN = 12000
N_TEST = 3000
WIDTHS = [256, 64, 32]
BATCH_SIZE = 256
EPOCHS = 8
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LAMBDA_OFFDIAG = 5e-3
PROBE_MAX_ITER = 2000


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist_numpy():
    rng = np.random.default_rng(SEED)
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())

    xtr = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    ytr = train_ds.targets.numpy()
    xte = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    yte = test_ds.targets.numpy()

    idx_tr = rng.choice(len(xtr), size=N_TRAIN, replace=False)
    idx_te = rng.choice(len(xte), size=N_TEST, replace=False)
    xtr, ytr = xtr[idx_tr], ytr[idx_tr]
    xte, yte = xte[idx_te], yte[idx_te]

    mu = xtr.mean(axis=0, keepdims=True)
    xtr = xtr - mu
    xte = xte - mu
    return xtr, ytr, xte, yte


def sample_pair_views(X, suite_name, seed):
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    mats = [np.eye(p, dtype=np.float32)] + test2.build_augmentation_suite(
        suite_name, h=28, w=28, rng=np.random.default_rng(seed)
    )
    idx1 = rng.integers(len(mats), size=X.shape[0])
    idx2 = rng.integers(len(mats), size=X.shape[0])

    x1 = np.empty_like(X)
    x2 = np.empty_like(X)
    for mat_idx, A in enumerate(mats):
        mask1 = idx1 == mat_idx
        mask2 = idx2 == mat_idx
        if np.any(mask1):
            x1[mask1] = X[mask1] @ A.T
        if np.any(mask2):
            x2[mask2] = X[mask2] @ A.T
    return x1, x2


def sample_same_class_pairs(X, y, seed):
    rng = np.random.default_rng(seed)
    x1 = X.copy()
    x2 = np.empty_like(X)
    classes = np.unique(y)

    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        if cls_idx.size == 1:
            x2[cls_idx] = X[cls_idx]
            continue

        choices = rng.integers(0, cls_idx.size, size=cls_idx.size)
        same = cls_idx[choices] == cls_idx
        if np.any(same):
            choices[same] = (choices[same] + 1) % cls_idx.size
        x2[cls_idx] = X[cls_idx[choices]]

    return x1, x2


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
    def __init__(self, input_dim, widths):
        super().__init__()
        dims = [input_dim] + list(widths)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(widths))]
        )

    def forward_to_layer_input(self, x, layer_idx):
        h = x
        for idx in range(layer_idx):
            h = torch.relu(self.layers[idx](h))
        return h

    def extract_features(self, x):
        h = x
        for layer in self.layers:
            h = torch.relu(layer(h))
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
                z1, z2 = torch.relu(pre1), torch.relu(pre2)

            loss = barlow_twins_loss(z1, z2, lambda_offdiag=lambda_offdiag)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
    return epoch_losses


def standardize_train_test(ztr, zte):
    mu = ztr.mean(axis=0, keepdims=True)
    std = ztr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-6, std, 1.0)
    return (ztr - mu) / std, (zte - mu) / std


def fit_linear_probe(ztr, ytr, zte, yte):
    clf_kwargs = {
        "max_iter": PROBE_MAX_ITER,
        "solver": "lbfgs",
    }
    if "multi_class" in signature(LogisticRegression).parameters:
        clf_kwargs["multi_class"] = "multinomial"
    clf = LogisticRegression(**clf_kwargs)
    clf.fit(ztr, ytr)
    return float((clf.predict(zte) == yte).mean())


def run_variant(
    suite_name,
    pair_source,
    loss_position,
    widths,
    batch_size,
    epochs,
    lr,
    momentum,
    weight_decay,
    lambda_offdiag,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr, ytr, xte, yte = load_mnist_numpy()
    if pair_source == "same-class":
        x1tr, x2tr = sample_same_class_pairs(xtr, ytr, seed=SEED + 101)
    else:
        x1tr, x2tr = sample_pair_views(xtr, suite_name, seed=SEED + 101)

    train_ds = TensorDataset(torch.from_numpy(x1tr), torch.from_numpy(x2tr))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = GreedyBarlowMLP(xtr.shape[1], widths).to(device)
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
        layer_stats.append(
            {
                "layer": layer_idx + 1,
                "final_epoch_loss": losses[-1],
                "loss_curve": losses,
            }
        )

    model.eval()
    with torch.no_grad():
        ztr = model.extract_features(torch.from_numpy(xtr).to(device)).cpu().numpy()
        zte = model.extract_features(torch.from_numpy(xte).to(device)).cpu().numpy()
    ztr, zte = standardize_train_test(ztr, zte)
    probe_acc = fit_linear_probe(ztr, ytr, zte, yte)
    return {
        "suite": suite_name,
        "pair_source": pair_source,
        "loss_position": loss_position,
        "widths": widths,
        "probe_accuracy": probe_acc,
        "layers": layer_stats,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="Greedy layerwise Barlow Twins MLP on MNIST.")
    parser.add_argument(
        "--suite",
        default="single-translation",
        choices=["image", "translation", "single-translation", "random-masking", "block-masking", "blurring"],
    )
    parser.add_argument(
        "--pair-source",
        default="suite",
        choices=["suite", "same-class"],
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-offdiag", type=float, default=LAMBDA_OFFDIAG)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for loss_position in ["pre", "post"]:
        results.append(
            run_variant(
                suite_name=args.suite,
                pair_source=args.pair_source,
                loss_position=loss_position,
                widths=WIDTHS,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                lambda_offdiag=args.lambda_offdiag,
            )
        )

    print(
        f"Greedy Barlow Twins MLP  |  suite={args.suite}  |  pair_source={args.pair_source}  |  widths={WIDTHS}"
    )
    for result in results:
        print(f"{result['loss_position']:>4s} activation loss | probe accuracy = {result['probe_accuracy']:.4f}")
        for layer in result["layers"]:
            print(f"  layer {layer['layer']} final loss = {layer['final_epoch_loss']:.4f}")

    if args.save_json is not None:
        args.save_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved json to {args.save_json}")


if __name__ == "__main__":
    main()
