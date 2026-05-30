import os
import time
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model import CascadedRiskAwareFilter

# Custom dataset for Simple Folder (e.g. HNH40K gesture image)
class SimpleFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for filename in sorted(os.listdir(root_dir)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                self.samples.append(os.path.join(root_dir, filename))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            # Return dummy black image if corrupt
            img = Image.new('RGB', (192, 160), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, path

# Custom dataset for HandGestureDataset (folders 0-19)
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in sorted(os.listdir(class_dir)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append(os.path.join(class_dir, filename))
                        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            img = Image.new('RGB', (192, 160), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, path

def evaluate_dataset(model, loader, device):
    model.eval()
    early_exit_count = 0
    hand_detected_count_05 = 0
    hand_detected_count_01 = 0
    total_valid = 0
    
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            # Run model step-by-step to track early exits
            # Stage 1: Fast Conv Stem
            features = model.stem(images)
            pooled_early = model.early_pool(features).flatten(1)
            early_logits = model.early_classifier(pooled_early)
            early_probs = torch.sigmoid(early_logits).squeeze(-1)
            
            # Stage 2: Structural Attention (run all for vectorization)
            final_logits = model._process_stage_two(features)
            final_probs = torch.sigmoid(final_logits).squeeze(-1)
            
            for i in range(images.size(0)):
                total_valid += 1
                e_prob = early_probs[i].item()
                f_prob = final_probs[i].item()
                
                # Simulation of dynamic exit during single-sample deployment:
                # If Stage 1 confidence < exit_threshold, exit immediately.
                if e_prob < model.exit_threshold:
                    early_exit_count += 1
                    pred_prob = e_prob
                else:
                    pred_prob = f_prob
                    
                if pred_prob >= 0.5:
                    hand_detected_count_05 += 1
                if pred_prob >= 0.1:
                    hand_detected_count_01 += 1
                    
    return {
        "total": total_valid,
        "early_exit_count": early_exit_count,
        "early_exit_rate": (early_exit_count / total_valid) * 100 if total_valid > 0 else 0,
        "recall_05": (hand_detected_count_05 / total_valid) * 100 if total_valid > 0 else 0,
        "recall_01": (hand_detected_count_01 / total_valid) * 100 if total_valid > 0 else 0
    }

def benchmark_paths(model, num_runs=500, warmup=100):
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, 192, 160)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            features = model.stem(dummy_input)
            pooled_early = model.early_pool(features).flatten(1)
            _ = model.early_classifier(pooled_early)
            _ = model._process_stage_two(features)
            
    # Measure Stage 1 Only (Early Exit Path)
    s1_latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            features = model.stem(dummy_input)
            pooled_early = model.early_pool(features).flatten(1)
            _ = model.early_classifier(pooled_early)
            s1_latencies.append((time.perf_counter() - start) * 1000) # ms
            
    # Measure Full Path (Stage 1 + Stage 2)
    full_latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            features = model.stem(dummy_input)
            pooled_early = model.early_pool(features).flatten(1)
            _ = model.early_classifier(pooled_early)
            _ = model._process_stage_two(features)
            full_latencies.append((time.perf_counter() - start) * 1000) # ms
            
    return {
        "cpu_early_exit_mean": np.mean(s1_latencies),
        "cpu_early_exit_std": np.std(s1_latencies),
        "cpu_full_mean": np.mean(full_latencies),
        "cpu_full_std": np.std(full_latencies)
    }

def main():
    checkpoint_path = '/data/selim_sarowar/HNH40K/weights/best_cascaded_model.pth'
    local_dataset_dir = '/data/selim_sarowar/HNH40K/HNH40K_Dataset/gesture image'
    public_dataset_dir = '/data/selim_sarowar/HNH40K/HandGestureDataset'
    results_dir = '/data/selim_sarowar/HNH40K/results'
    
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading Cascaded Risk-Aware Filter model from {checkpoint_path}...")
    model = CascadedRiskAwareFilter(exit_threshold=0.1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Define standard input transform (same as HNH40K training/val)
    transform = transforms.Compose([
        transforms.Resize((192, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 1. Benchmark CPU latency of paths
    print("Profiling model path latencies on CPU (500 runs)...")
    lat_results = benchmark_paths(model)
    print(f"  CPU Early Exit Path: {lat_results['cpu_early_exit_mean']:.3f} +- {lat_results['cpu_early_exit_std']:.3f} ms")
    print(f"  CPU Full Path:       {lat_results['cpu_full_mean']:.3f} +- {lat_results['cpu_full_std']:.3f} ms")
    
    # Move model back to device
    model.to(device)
    
    # 2. Evaluate Local Gesture Dataset
    print("\nEvaluating Local High-Resolution 'gesture image' dataset...")
    local_ds = SimpleFolderDataset(local_dataset_dir, transform=transform)
    local_loader = DataLoader(local_ds, batch_size=128, shuffle=False, num_workers=4)
    local_metrics = evaluate_dataset(model, local_loader, device)
    
    # Compute dynamic latency based on exit rate
    local_dynamic_lat = (
        (local_metrics['early_exit_rate'] / 100) * lat_results['cpu_early_exit_mean'] +
        (1 - local_metrics['early_exit_rate'] / 100) * lat_results['cpu_full_mean']
    )
    local_metrics['dynamic_latency_cpu'] = local_dynamic_lat
    
    print(f"  Total Images: {local_metrics['total']}")
    print(f"  Recall (>=0.5): {local_metrics['recall_05']:.2f}%")
    print(f"  Recall (>=0.1): {local_metrics['recall_01']:.2f}%")
    print(f"  Early-Exit Rate: {local_metrics['early_exit_rate']:.2f}%")
    print(f"  Dynamic CPU Latency: {local_dynamic_lat:.3f} ms")
    
    # 3. Evaluate Public Low-Resolution HandGestureDataset
    print("\nEvaluating Public Low-Resolution 'HandGestureDataset' (20 classes)...")
    public_ds = HandGestureDataset(public_dataset_dir, transform=transform)
    public_loader = DataLoader(public_ds, batch_size=128, shuffle=False, num_workers=4)
    public_metrics = evaluate_dataset(model, public_loader, device)
    
    public_dynamic_lat = (
        (public_metrics['early_exit_rate'] / 100) * lat_results['cpu_early_exit_mean'] +
        (1 - public_metrics['early_exit_rate'] / 100) * lat_results['cpu_full_mean']
    )
    public_metrics['dynamic_latency_cpu'] = public_dynamic_lat
    
    print(f"  Total Images: {public_metrics['total']}")
    print(f"  Recall (>=0.5): {public_metrics['recall_05']:.2f}%")
    print(f"  Recall (>=0.1): {public_metrics['recall_01']:.2f}%")
    print(f"  Early-Exit Rate: {public_metrics['early_exit_rate']:.2f}%")
    print(f"  Dynamic CPU Latency: {public_dynamic_lat:.3f} ms")
    
    # Save JSON report
    report = {
        "local_dataset": {
            "name": "Local Gesture Image (High-Res)",
            "resolution": "3072x3072",
            **local_metrics
        },
        "public_dataset": {
            "name": "HandGestureDataset (Low-Res)",
            "resolution": "50x50",
            **public_metrics
        },
        "latencies": lat_results
    }
    
    json_path = os.path.join(results_dir, 'ood_evaluation_results.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved JSON report to {json_path}")
    
    # Generate LaTeX tables
    latex_path = os.path.join(results_dir, 'ood_evaluation_results.tex')
    
    latex_content = f"""% LaTeX code for paper generalization performance tables
% Copy and paste into your paper's LaTeX document.

\\begin{{table}}[h!]
\\centering
\\caption{{Generalization performance and computational efficiency of the Cascaded Risk-Aware Filter on out-of-distribution (OOD) hand gesture datasets. Early-exit rate reflects the proportion of frames processed entirely by the Stage 1 stem, bypassing Stage 2 structural self-attention.}}
\\label{{tab:ood_generalization}}
\\begin{{tabular}}{{lccccc}}
\\hline
\\textbf{{OOD Dataset}} & \\textbf{{Source Resolution}} & \\textbf{{Samples}} & \\textbf{{Recall ($\\ge 0.5$)}} & \\textbf{{Early-Exit Rate}} & \\textbf{{Dynamic Latency (CPU)}} \\\\
\\hline
Local Gesture Image & $3072 \\times 3072$ & {local_metrics['total']} & {local_metrics['recall_05']:.2f}\\% & {local_metrics['early_exit_rate']:.2f}\\% & {local_dynamic_lat:.2f} ms \\\\
HandGestureDataset & $50 \\times 50$ & {public_metrics['total']} & {public_metrics['recall_05']:.2f}\\% & {public_metrics['early_exit_rate']:.2f}\\% & {public_dynamic_lat:.2f} ms \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

% Detailed computational profiling table
\\begin{{table}}[h!]
\\centering
\\caption{{Computational profiling of the Cascaded Risk-Aware Filter on edge CPU device. Latency is measured per-sample (batch size = 1).}}
\\label{{tab:latency_profiling}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Inference Path}} & \\textbf{{Avg. Latency (CPU)}} & \\textbf{{Throughput (CPU)}} \\\\
\\hline
Stage 1 Only (Early Exit) & {lat_results['cpu_early_exit_mean']:.2f} $\\pm$ {lat_results['cpu_early_exit_std']:.2f} ms & {1000 / lat_results['cpu_early_exit_mean']:.1f} FPS \\\\
Full Path (Stage 1 + Stage 2) & {lat_results['cpu_full_mean']:.2f} $\\pm$ {lat_results['cpu_full_std']:.2f} ms & {1000 / lat_results['cpu_full_mean']:.1f} FPS \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    with open(latex_path, 'w') as f:
        f.write(latex_content)
    print(f"Saved LaTeX snippet to {latex_path}")
    print("\n--- LATEX TABLE SNIPPET ---")
    print(latex_content)
    print("---------------------------")

if __name__ == "__main__":
    main()
