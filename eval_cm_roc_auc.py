"""
Experiment 1 — Confusion Matrix, ROC curve, AUC, and per-class metrics
for the HNH40K Cascaded Risk-Aware Filter.

Outputs (under results/exp1/):
    confusion_matrix.pdf / .png
    roc_curve.pdf / .png
    metrics.json
    metrics.csv
    per_threshold.csv
"""
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
)

from dataset import HNH40KBinaryDataset
from model import CascadedRiskAwareFilter

CLASS_NAMES = ["Non-Hand", "Hand"]


def collect_predictions(model, loader, device):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def plot_confusion_matrix(cm, out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"size": 14},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title("HNH40K — Confusion Matrix (Validation)", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"confusion_matrix.{ext}", dpi=200)
    plt.close(fig)


def plot_roc_pr(labels, probs, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.plot(fpr, tpr, color="C0", lw=2.0, label=f"CRAF (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.0, linestyle="--", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(recall, precision, color="C2", lw=2.0, label=f"CRAF (AP = {ap:.4f})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    fig.suptitle("HNH40K — Discrimination Quality", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"roc_curve.{ext}", dpi=200)
    plt.close(fig)

    return roc_auc, ap


def threshold_sweep(labels, probs, out_dir):
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
        })
    import csv
    keys = list(rows[0].keys())
    with open(out_dir / "per_threshold.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/selim_sarowar/HNH40K/HNH40K_Dataset")
    parser.add_argument("--checkpoint", default="/data/selim_sarowar/HNH40K/weights/best_cascaded_model.pth")
    parser.add_argument("--out_dir", default="/data/selim_sarowar/HNH40K/results/exp1_cm_roc_auc")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp1] device={device}  checkpoint={args.checkpoint}")

    transform = transforms.Compose([
        transforms.Resize((192, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full = HNH40KBinaryDataset(root_dir=args.data_dir, transform=transform)
    n = len(full)
    n_train = int(0.8 * n)
    n_val = n - n_train
    _, val = torch.utils.data.random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[exp1] val size = {len(val)}")
    loader = DataLoader(val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = CascadedRiskAwareFilter(exit_threshold=0.1).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    labels, probs = collect_predictions(model, loader, device)
    preds = (probs >= 0.5).astype(int)
    labels_int = labels.astype(int)

    cm = confusion_matrix(labels_int, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "n_samples": int(len(labels)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(labels_int, preds)),
        "precision_hand": float(precision_score(labels_int, preds, zero_division=0)),
        "recall_hand": float(recall_score(labels_int, preds, zero_division=0)),
        "f1_hand": float(f1_score(labels_int, preds, zero_division=0)),
        "precision_nonhand": float(precision_score(1 - labels_int, 1 - preds, zero_division=0)),
        "recall_nonhand": float(recall_score(1 - labels_int, 1 - preds, zero_division=0)),
        "f1_nonhand": float(f1_score(1 - labels_int, 1 - preds, zero_division=0)),
        "macro_f1": float(f1_score(labels_int, preds, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_int, preds)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

    plot_confusion_matrix(cm, out_dir)
    roc_auc, ap = plot_roc_pr(labels_int, probs, out_dir)
    metrics["roc_auc"] = float(roc_auc)
    metrics["average_precision"] = float(ap)

    rows = threshold_sweep(labels_int, probs, out_dir)
    best = max(rows, key=lambda r: r["f1"])
    metrics["best_threshold_by_f1"] = best

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    import csv
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, dict):
                continue
            writer.writerow([k, v])

    report = classification_report(
        labels_int, preds, target_names=CLASS_NAMES, digits=4
    )
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    print("\n=== HNH40K Exp 1 — Discrimination Metrics ===")
    print(report)
    print(f"ROC AUC          : {roc_auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"MCC              : {metrics['mcc']:.4f}")
    print(f"FPR / FNR        : {metrics['fpr']:.4f} / {metrics['fnr']:.4f}")
    print(f"Best F1 @ thr={best['threshold']:.2f}: F1={best['f1']:.4f}  Acc={best['accuracy']:.4f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
