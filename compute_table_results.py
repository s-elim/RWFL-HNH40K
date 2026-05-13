import argparse
import os
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from dataset import HNH40KBinaryDataset
from model import CascadedRiskAwareFilter


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    roc_auc: Optional[float]
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class Profile:
    params_m: float
    flops_g: Optional[float]
    size_mb: float
    vram_gb: Optional[float]
    gpu_ms: Optional[Tuple[float, float]]
    cpu_ms: Tuple[float, float]


def build_transform(image_size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_model(checkpoint: str, device: torch.device) -> CascadedRiskAwareFilter:
    model = CascadedRiskAwareFilter(exit_threshold=0.1).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def stratified_subset_indices(dataset: HNH40KBinaryDataset, total: int, seed: int) -> List[int]:
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    pos_indices = [i for i, y in enumerate(labels) if y == 1.0]
    neg_indices = [i for i, y in enumerate(labels) if y == 0.0]

    random.Random(seed).shuffle(pos_indices)
    random.Random(seed + 1).shuffle(neg_indices)

    pos_frac = len(pos_indices) / max(len(labels), 1)
    pos_count = int(round(total * pos_frac))
    neg_count = total - pos_count

    pos_count = min(pos_count, len(pos_indices))
    neg_count = min(neg_count, len(neg_indices))

    subset = pos_indices[:pos_count] + neg_indices[:neg_count]
    random.Random(seed + 2).shuffle(subset)
    return subset


def compute_auc(scores: List[float], labels: List[int]) -> Optional[float]:
    if not scores:
        return None
    pos_scores = [(s, l) for s, l in zip(scores, labels)]
    n_pos = sum(l == 1 for _, l in pos_scores)
    n_neg = len(pos_scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranked = sorted(pos_scores, key=lambda x: x[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(ranked):
        score = ranked[idx][0]
        start = idx
        while idx < len(ranked) and ranked[idx][0] == score:
            idx += 1
        end = idx
        avg_rank = (start + 1 + end) / 2.0
        for _, label in ranked[start:end]:
            if label == 1:
                rank_sum += avg_rank
    auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    tp = tn = fp = fn = 0
    scores: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= 0.5).float()

            for i in range(len(batch_labels)):
                label = int(batch_labels[i].item())
                pred = int(preds[i].item())
                score = float(probs[i].item())
                scores.append(score)
                labels.append(label)

                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc = compute_auc(scores, labels)

    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        roc_auc=roc_auc,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_model_size_mb(model: torch.nn.Module) -> float:
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)


def estimate_flops(model: torch.nn.Module, device: torch.device, image_size: Tuple[int, int]) -> Optional[float]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None

    input_tensor = torch.randn(1, 3, image_size[0], image_size[1], device=device)
    with FlopCounterMode(model, display=False) as flops:
        model(input_tensor)
    total_flops = flops.get_total_flops()
    return total_flops / 1e9 if total_flops is not None else None


def profile_latency(model: torch.nn.Module, device: torch.device, image_size: Tuple[int, int],
                    iters: int = 100, warmup: int = 20) -> Tuple[Tuple[float, float], Optional[float]]:
    batch = torch.randn(1, 3, image_size[0], image_size[1], device=device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times: List[float] = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            for _ in range(iters):
                start.record()
                _ = model(batch)
                end.record()
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(end))
        vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        with torch.inference_mode():
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = model(batch)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)
        vram_gb = None

    mean_ms = float(sum(times) / len(times))
    std_ms = float(math.sqrt(sum((t - mean_ms) ** 2 for t in times) / len(times)))
    return (mean_ms, std_ms), vram_gb


def profile_cpu_latency(model: torch.nn.Module, image_size: Tuple[int, int],
                        iters: int = 50, warmup: int = 10) -> Tuple[float, float]:
    device = torch.device("cpu")
    model_cpu = model.to(device)
    batch = torch.randn(1, 3, image_size[0], image_size[1], device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model_cpu(batch)

    times: List[float] = []
    with torch.inference_mode():
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model_cpu(batch)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(sum(times) / len(times))
    std_ms = float(math.sqrt(sum((t - mean_ms) ** 2 for t in times) / len(times)))
    return mean_ms, std_ms


def compute_scale_metrics(root_dir: str, checkpoint: str, device: torch.device,
                         total_samples: int, batch_size: int, seed: int,
                         image_size: Tuple[int, int]) -> Tuple[Metrics, int]:
    transform = build_transform(image_size)
    dataset = HNH40KBinaryDataset(root_dir=root_dir, transform=transform)

    indices = stratified_subset_indices(dataset, total_samples, seed)
    subset = Subset(dataset, indices)
    if len(subset) < total_samples:
        print(f"Warning: subset size {len(subset)} < requested {total_samples}")

    eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = load_model(checkpoint, device)
    metrics = evaluate_model(model, eval_loader, device)
    return metrics, len(subset)


def compute_profile(checkpoint: str, device: torch.device, image_size: Tuple[int, int]) -> Profile:
    model = load_model(checkpoint, device)
    params = count_params(model)
    params_m = params / 1e6
    size_mb = count_model_size_mb(model)
    flops_g = estimate_flops(model, device, image_size)

    gpu_stats = None
    vram_gb = None
    if device.type == "cuda":
        gpu_stats, vram_gb = profile_latency(model, device, image_size)

    cpu_stats = profile_cpu_latency(model, image_size)

    return Profile(
        params_m=params_m,
        flops_g=flops_g,
        size_mb=size_mb,
        vram_gb=vram_gb,
        gpu_ms=gpu_stats,
        cpu_ms=cpu_stats,
    )


def write_metrics_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute HNH40K table metrics")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (192, 160)

    scales = [5000, 10000, 20000, 40000]
    metrics_rows = []
    metrics_json: Dict[str, Dict[str, object]] = {}

    for total in scales:
        metrics, eval_count = compute_scale_metrics(
            root_dir=args.data_dir,
            checkpoint=args.checkpoint,
            device=device,
            total_samples=total,
            batch_size=args.batch_size,
            seed=args.seed + total,
            image_size=image_size,
        )
        row = {
            "scale": total,
            "eval_count": eval_count,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "fpr": metrics.fpr,
            "roc_auc": metrics.roc_auc,
            "tp": metrics.tp,
            "tn": metrics.tn,
            "fp": metrics.fp,
            "fn": metrics.fn,
        }
        metrics_rows.append(row)
        metrics_json[str(total)] = row

    profiles_rows = []
    for size in [(128, 128), (256, 256)]:
        profile = compute_profile(args.checkpoint, device, size)
        row = {
            "input_size": f"{size[0]}x{size[1]}",
            "params_m": profile.params_m,
            "flops_g": profile.flops_g,
            "size_mb": profile.size_mb,
            "vram_gb": profile.vram_gb,
            "gpu_ms_mean": profile.gpu_ms[0] if profile.gpu_ms else None,
            "gpu_ms_std": profile.gpu_ms[1] if profile.gpu_ms else None,
            "cpu_ms_mean": profile.cpu_ms[0],
            "cpu_ms_std": profile.cpu_ms[1],
        }
        profiles_rows.append(row)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "table_metrics.csv")
    metrics_json_path = os.path.join(out_dir, "table_metrics.json")
    profile_path = os.path.join(out_dir, "table_profiles.csv")
    profile_json_path = os.path.join(out_dir, "table_profiles.json")

    write_metrics_csv(metrics_path, metrics_rows)
    write_metrics_csv(profile_path, profiles_rows)

    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    with open(profile_json_path, "w") as f:
        json.dump(profiles_rows, f, indent=2)

    print("Metrics written to:")
    print(f"- {metrics_path}")
    print(f"- {metrics_json_path}")
    print("Profiles written to:")
    print(f"- {profile_path}")
    print(f"- {profile_json_path}")


if __name__ == "__main__":
    main()
