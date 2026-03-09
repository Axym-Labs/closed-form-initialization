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
import closed_form_barlow_twins_cifar as cfbt_cifar
from project_paths import resolve_json_path


SEED = 7
WIDTHS = [256, 64, 32]
BATCH_SIZE = 256
EPOCHS = 100
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LAMBDA_OFFDIAG = 5e-3
BT_WEIGHT = 0.1
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


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, widths, num_classes, activation):
        super().__init__()
        dims = [input_dim] + list(widths)
        self.activation = activation
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(len(widths))]
        )
        self.head = nn.Linear(widths[-1], num_classes, bias=True)

    def encode(self, x):
        h = x
        for layer in self.hidden:
            h = apply_activation_torch(layer(h), self.activation)
        return h

    def forward(self, x):
        h = self.encode(x)
        logits = self.head(h)
        return h, logits


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


def evaluate(model, xtr, ytr, xte, yte, device):
    model.eval()
    with torch.no_grad():
        ztr, logits_tr = model(torch.from_numpy(xtr).float().to(device))
        zte, logits_te = model(torch.from_numpy(xte).float().to(device))
        ztr = ztr.cpu().numpy()
        zte = zte.cpu().numpy()
        pred = logits_te.argmax(dim=1).cpu().numpy()
    probe_tr, probe_te = cfbt.standardize_train_test(ztr, zte)
    probe_acc = fit_linear_probe(probe_tr, ytr, probe_te, yte)
    classifier_acc = float((pred == yte).mean())
    return probe_acc, classifier_acc


def run_experiment(
    dataset_name,
    suite_name,
    mode,
    widths,
    activation,
    batch_size,
    epochs,
    lr,
    momentum,
    weight_decay,
    lambda_offdiag,
    bt_weight,
    n_train,
    n_test,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xtr_img, ytr, xte_img, yte = cfbt_cifar.load_cifar_numpy(dataset_name, n_train=n_train, n_test=n_test, seed=SEED)
    xtr = cfbt_cifar.images_to_flat(xtr_img)
    xte = cfbt_cifar.images_to_flat(xte_img)
    x1tr, x2tr = cfbt_cifar.sample_pair_views(xtr_img, suite_name, seed=SEED + 101)

    num_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    model = MLPClassifier(xtr.shape[1], widths, num_classes=num_classes, activation=activation).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    ce_loss = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.from_numpy(xtr).float(),
        torch.from_numpy(ytr).long(),
        torch.from_numpy(x1tr).float(),
        torch.from_numpy(x2tr).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    epoch_stats = []
    for _ in range(epochs):
        model.train()
        ce_losses = []
        bt_losses = []
        total_losses = []
        for xb, yb, x1b, x2b in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            x1b = x1b.to(device)
            x2b = x2b.to(device)

            feat_base, logits = model(xb)
            loss_sup = ce_loss(logits, yb)
            loss = loss_sup
            loss_bt = torch.tensor(0.0, device=device)

            if mode == "supervised+bt":
                feat1, _ = model(x1b)
                feat2, _ = model(x2b)
                loss_bt = barlow_twins_loss(feat1, feat2, lambda_offdiag=lambda_offdiag)
                loss = loss_sup + bt_weight * loss_bt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ce_losses.append(float(loss_sup.item()))
            bt_losses.append(float(loss_bt.item()))
            total_losses.append(float(loss.item()))

        epoch_stats.append(
            {
                "ce_loss": float(np.mean(ce_losses)),
                "bt_loss": float(np.mean(bt_losses)),
                "total_loss": float(np.mean(total_losses)),
            }
        )

    probe_acc, classifier_acc = evaluate(model, xtr, ytr, xte, yte, device)
    return {
        "dataset": dataset_name,
        "suite": suite_name,
        "mode": mode,
        "activation": activation,
        "widths": widths,
        "epochs": epochs,
        "n_train": n_train,
        "n_test": n_test,
        "probe_accuracy": probe_acc,
        "classifier_accuracy": classifier_acc,
        "epoch_stats": epoch_stats,
        "bt_weight": bt_weight,
        "lambda_offdiag": lambda_offdiag,
        "device": str(device),
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end supervised and supervised+BT MLP baselines on CIFAR.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--suite", default="single-translation", choices=["single-translation", "block-masking"])
    parser.add_argument("--mode", choices=["supervised", "supervised+bt", "both"], default="both")
    parser.add_argument("--activation", choices=["relu", "tanh", "leaky-relu", "identity"], default="relu")
    parser.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda-offdiag", type=float, default=LAMBDA_OFFDIAG)
    parser.add_argument("--bt-weight", type=float, default=BT_WEIGHT)
    parser.add_argument("--n-train", type=int, default=cfbt_cifar.N_TRAIN)
    parser.add_argument("--n-test", type=int, default=cfbt_cifar.N_TEST)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    modes = ["supervised", "supervised+bt"] if args.mode == "both" else [args.mode]
    results = []
    for mode in modes:
        results.append(
            run_experiment(
                dataset_name=args.dataset,
                suite_name=args.suite,
                mode=mode,
                widths=args.widths,
                activation=args.activation,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                lambda_offdiag=args.lambda_offdiag,
                bt_weight=args.bt_weight,
                n_train=args.n_train,
                n_test=args.n_test,
            )
        )

    print(
        f"Backprop CIFAR baselines  |  dataset={args.dataset}  |  suite={args.suite}  |  activation={args.activation}  |  widths={args.widths}"
    )
    for result in results:
        print(
            f"{result['mode']:>13s} | classifier acc = {result['classifier_accuracy']:.4f} | probe acc = {result['probe_accuracy']:.4f} | final total loss = {result['epoch_stats'][-1]['total_loss']:.4f}"
        )

    if args.save_json is not None:
        output_path = resolve_json_path(args.save_json)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved json to {output_path}")


if __name__ == "__main__":
    main()
