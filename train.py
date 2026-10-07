"""Train ParkiTrace.

Windows example:
    python train.py --data dataset --epochs 15

For a CPU-only laptop, start with:
    python train.py --data dataset --epochs 5 --freeze-backbones

The project saves the best validation model plus accuracy/precision/recall/F1.
"""
import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

from parkitrace_model import CLASSES, MEAN, STD, ParkiTraceNet


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download ImageNet weights.")
    parser.add_argument("--freeze-backbones", action="store_true",
                        help="Train only the fusion/classification head; recommended for CPU.")
    args = parser.parse_args()

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.isdir(args.data):
        raise SystemExit(f"Dataset folder not found: {args.data}")
    full = datasets.ImageFolder(args.data)
    if full.classes != CLASSES:
        raise SystemExit(
            f"Expected exactly these dataset folders: {CLASSES}. Found: {full.classes}"
        )
    counts = np.bincount(full.targets)
    if len(counts) != 2 or np.min(counts) < 2:
        raise SystemExit("Each class needs at least 2 images.")

    indices = np.arange(len(full))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=full.targets, random_state=42
    )

    train_tf = T.Compose([
        T.Resize((240, 240)),
        T.RandomResizedCrop(224, scale=(0.85, 1.0)),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.15),
        T.ToTensor(),
        T.Normalize(MEAN.tolist(), STD.tolist()),
    ])
    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN.tolist(), STD.tolist()),
    ])

    train_ds = Subset(datasets.ImageFolder(args.data, transform=train_tf), train_idx)
    val_ds = Subset(datasets.ImageFolder(args.data, transform=val_tf), val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=0)

    net = ParkiTraceNet(
        pretrained=not args.no_pretrained,
        freeze_backbones=args.freeze_backbones,
    ).to(device)

    trainable = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    os.makedirs("model", exist_ok=True)

    best_acc = -1.0
    best_state = None
    best_pred, best_true = None, None

    for epoch in range(1, args.epochs + 1):
        net.train()
        if args.freeze_backbones:
            net.cnn.eval()
            net.vit.eval()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        net.eval()
        preds, true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = net(x.to(device))
                preds.extend(logits.argmax(1).cpu().tolist())
                true.extend(y.tolist())

        acc = float(np.mean(np.asarray(preds) == np.asarray(true)))
        loss = running_loss / max(1, len(train_ds))
        print(f"epoch {epoch}/{args.epochs}  train_loss={loss:.4f}  val_acc={acc:.4f}")

        if acc >= best_acc:
            best_acc = acc
            best_pred, best_true = preds[:], true[:]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save({"state": best_state, "classes": CLASSES}, "model/parkitrace.pt")

    report = classification_report(
        best_true, best_pred, target_names=CLASSES, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(best_true, best_pred)
    metrics = {
        "accuracy": float(best_acc),
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1": float(report["weighted avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "device": str(device),
        "epochs": args.epochs,
        "freeze_backbones": args.freeze_backbones,
    }
    with open("model/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(2), CLASSES)
    ax.set_yticks(range(2), CLASSES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ParkiTrace validation confusion matrix")
    fig.tight_layout()
    fig.savefig("model/confusion_matrix.png", dpi=150)
    plt.close(fig)

    print("\nBest validation metrics:")
    print(classification_report(best_true, best_pred, target_names=CLASSES, zero_division=0))
    print("Saved: model/parkitrace.pt")
    print("Saved: model/metrics.json")
    print("Saved: model/confusion_matrix.png")


if __name__ == "__main__":
    main()
