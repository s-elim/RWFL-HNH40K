"""
Experiment 2 — Perturbation Robustness Analysis (ImageNet-C style)
for the HNH40K Cascaded Risk-Aware Filter.

Perturbation types (7):
    Gaussian Noise, Gaussian Blur, Brightness, Contrast,
    JPEG Compression, Occlusion (random patch), Rotation

Each perturbation is tested at 5 severity levels.

Outputs (under results/exp2/):
    perturbation_table.csv
    perturbation_table.json
    perturbation_lineplot.pdf / .png
    perturbation_heatmap.pdf / .png
    perturbation_bar.pdf / .png
"""
import os
import io
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from dataset import HNH40KBinaryDataset
from model import CascadedRiskAwareFilter


# ---------------------------------------------------------------------------
# Perturbation functions — operate on PIL Images, return PIL Images
# ---------------------------------------------------------------------------

def gaussian_noise(img, severity):
    """Add Gaussian noise. severity 1-5 → sigma 5..40."""
    sigma = [5, 12, 20, 30, 40][severity - 1]
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def gaussian_blur(img, severity):
    """Apply Gaussian blur. severity 1-5 → radius 1..7."""
    radius = [1, 2, 3, 5, 7][severity - 1]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def brightness(img, severity):
    """Shift brightness. severity 1-5 → factor 0.85..0.3 (darker)."""
    factor = [0.85, 0.7, 0.55, 0.4, 0.3][severity - 1]
    arr = np.array(img, dtype=np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def contrast(img, severity):
    """Reduce contrast by blending with mean-gray."""
    factor = [0.85, 0.7, 0.55, 0.4, 0.25][severity - 1]
    arr = np.array(img, dtype=np.float32)
    mean = arr.mean()
    arr = (arr - mean) * factor + mean
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg_compression(img, severity):
    """Re-encode with progressively lower JPEG quality."""
    quality = [50, 30, 15, 8, 3][severity - 1]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def occlusion(img, severity):
    """Black-out a random square patch. severity → patch fraction 10%-50%."""
    frac = [0.10, 0.20, 0.30, 0.40, 0.50][severity - 1]
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    ph, pw = int(h * frac), int(w * frac)
    y = np.random.randint(0, max(1, h - ph))
    x = np.random.randint(0, max(1, w - pw))
    arr[y:y + ph, x:x + pw] = 0
    return Image.fromarray(arr)


def rotation(img, severity):
    """Rotate with black fill. severity 1-5 → angle 5..45."""
    angle = [5, 12, 20, 30, 45][severity - 1]
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0))


PERTURBATIONS = OrderedDict([
    ("Gaussian Noise", gaussian_noise),
    ("Gaussian Blur", gaussian_blur),
    ("Brightness", brightness),
    ("Contrast", contrast),
    ("JPEG Compression", jpeg_compression),
    ("Occlusion", occlusion),
    ("Rotation", rotation),
])


# ---------------------------------------------------------------------------
# Dataset wrapper that applies a perturbation to each sample
# ---------------------------------------------------------------------------

class PerturbedDataset(Dataset):
    """Wraps an HNH40KBinaryDataset (or Subset), applies perturbation on PIL,
    then applies the standard transform (resize + normalize)."""

    def __init__(self, base_dataset, perturb_fn, severity, transform):
        self.base = base_dataset
        self.perturb_fn = perturb_fn
        self.severity = severity
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # base_dataset may be a Subset wrapping HNH40KBinaryDataset
        # We need the raw PIL image before transforms
        if hasattr(self.base, "dataset"):
            # It's a Subset
            real_idx = self.base.indices[idx]
            img_path, label_val, weight_val = self.base.dataset.samples[real_idx]
        else:
            img_path, label_val, weight_val = self.base.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (192, 160), (0, 0, 0))

        img = self.perturb_fn(img, self.severity)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label_val, dtype=torch.float32)
        return img, label


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, loader, device):
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)
    labels_int = labels.astype(int)
    return {
        "accuracy": float(accuracy_score(labels_int, preds)),
        "f1": float(f1_score(labels_int, preds, zero_division=0)),
        "precision": float(precision_score(labels_int, preds, zero_division=0)),
        "recall": float(recall_score(labels_int, preds, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_lineplot(results, clean_metrics, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    severities = list(range(1, 6))
    for name in PERTURBATIONS:
        accs = [results[name][s]["accuracy"] for s in severities]
        ax.plot(severities, accs, marker="o", linewidth=2, label=name)
    ax.axhline(clean_metrics["accuracy"], color="black", linestyle="--",
               linewidth=1.5, label=f"Clean ({clean_metrics['accuracy']:.3f})")
    ax.set_xlabel("Severity Level", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("HNH40K — Perturbation Robustness", fontsize=13)
    ax.set_xticks(severities)
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"perturbation_lineplot.{ext}", dpi=200)
    plt.close(fig)


def plot_heatmap(results, out_dir):
    names = list(PERTURBATIONS.keys())
    severities = list(range(1, 6))
    data = np.array([[results[n][s]["accuracy"] for s in severities] for n in names])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.3, vmax=1.0,
                xticklabels=[f"Sev {s}" for s in severities],
                yticklabels=names, ax=ax, annot_kws={"size": 10})
    ax.set_title("Accuracy under Perturbation (severity 1-5)", fontsize=13)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"perturbation_heatmap.{ext}", dpi=200)
    plt.close(fig)


def plot_bar(results, clean_metrics, out_dir):
    """Bar chart: accuracy drop at severity 5 relative to clean."""
    names = list(PERTURBATIONS.keys())
    drops = [clean_metrics["accuracy"] - results[n][5]["accuracy"] for n in names]
    colors = ["#e74c3c" if d > 0.1 else "#f39c12" if d > 0.03 else "#27ae60" for d in drops]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, drops, color=colors)
    ax.set_xlabel("Accuracy Drop (Clean − Severity 5)", fontsize=11)
    ax.set_title("HNH40K — Vulnerability Profile", fontsize=13)
    for bar, drop in zip(bars, drops):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{drop:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, max(drops) * 1.25 if max(drops) > 0 else 0.1)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"perturbation_bar.{ext}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/selim_sarowar/HNH40K/HNH40K_Dataset")
    parser.add_argument("--checkpoint", default="/data/selim_sarowar/HNH40K/weights/best_cascaded_model.pth")
    parser.add_argument("--out_dir", default="/data/selim_sarowar/HNH40K/results/exp2_perturbation")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[exp2] device={device}  checkpoint={args.checkpoint}")

    # We need raw PIL images → perturbation → then transform
    # So load dataset without transform for the split, then wrap
    raw_dataset = HNH40KBinaryDataset(root_dir=args.data_dir, transform=None)
    n = len(raw_dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    _, val_subset = torch.utils.data.random_split(
        raw_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[exp2] val size = {len(val_subset)}")

    normalize_transform = transforms.Compose([
        transforms.Resize((192, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = CascadedRiskAwareFilter(exit_threshold=0.1).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # --- Clean baseline ---
    clean_ds = PerturbedDataset(val_subset, lambda img, _: img, 1, normalize_transform)
    clean_loader = DataLoader(clean_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    clean_metrics = evaluate(model, clean_loader, device)
    print(f"[clean] acc={clean_metrics['accuracy']:.4f}  f1={clean_metrics['f1']:.4f}")

    # --- Perturbation sweep ---
    results = {}
    severities = list(range(1, 6))
    for name, fn in PERTURBATIONS.items():
        results[name] = {}
        for sev in severities:
            np.random.seed(args.seed + sev)  # reproducible noise/occlusion
            ds = PerturbedDataset(val_subset, fn, sev, normalize_transform)
            loader = DataLoader(ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
            m = evaluate(model, loader, device)
            results[name][sev] = m
            delta = clean_metrics["accuracy"] - m["accuracy"]
            print(f"  {name:20s}  sev={sev}  acc={m['accuracy']:.4f}  "
                  f"f1={m['f1']:.4f}  drop={delta:+.4f}")

    # --- Save table ---
    import csv
    rows = []
    for name in PERTURBATIONS:
        for sev in severities:
            row = {"perturbation": name, "severity": sev}
            row.update(results[name][sev])
            row["acc_drop"] = clean_metrics["accuracy"] - results[name][sev]["accuracy"]
            rows.append(row)
    keys = list(rows[0].keys())
    with open(out_dir / "perturbation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    full_output = {"clean": clean_metrics, "perturbations": {}}
    for name in PERTURBATIONS:
        full_output["perturbations"][name] = {
            str(s): results[name][s] for s in severities
        }
    with open(out_dir / "perturbation_table.json", "w") as f:
        json.dump(full_output, f, indent=2)

    # --- Plots ---
    plot_lineplot(results, clean_metrics, out_dir)
    plot_heatmap(results, out_dir)
    plot_bar(results, clean_metrics, out_dir)

    # --- Summary ---
    print("\n=== HNH40K Exp 2 — Perturbation Robustness Summary ===")
    print(f"Clean Accuracy: {clean_metrics['accuracy']:.4f}")
    print(f"{'Perturbation':20s}  {'Sev1':>7s}  {'Sev3':>7s}  {'Sev5':>7s}  {'Drop@5':>7s}")
    print("-" * 60)
    for name in PERTURBATIONS:
        s1 = results[name][1]["accuracy"]
        s3 = results[name][3]["accuracy"]
        s5 = results[name][5]["accuracy"]
        drop = clean_metrics["accuracy"] - s5
        print(f"{name:20s}  {s1:7.4f}  {s3:7.4f}  {s5:7.4f}  {drop:+7.4f}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
