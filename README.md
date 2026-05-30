# HNH40K: Cascaded Risk-Aware Filter for Robust Hand Detection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Venue: BMVC 2026](https://img.shields.io/badge/Venue-BMVC%202026-blue.svg)]()

Official PyTorch implementation of the **Cascaded Risk-Aware Filter** and the **HNH40K Dataset**. 

This repository provides an ultra-efficient, highly robust front-end hand vs. non-hand filter. It is designed to ruthlessly suppress false positives at the edge, serving as a reliable gating mechanism before invoking computationally heavy Vision-Language-Action (VLA) models, World Models, or 5D robot manipulation frameworks.

## 🧠 Model Architecture

Standard edge CNNs struggle with visually similar distractors (faces, printed hands, toys), while global context models (Transformers/SSMs) are too computationally heavy for early-stage filtering. 

Our architecture solves this via **Computational Asymmetry**:

1. **Stage 1: The Fast Rejector (Stem)** A truncated `MobileNetV3-Small` stem processes all frames. It extracts morphological features and dynamically exits early on low-risk backgrounds (e.g., landscapes, blank walls), bypassing heavy computation for ~60-70% of standard inference frames.
   
2. **Stage 2: The Structural Scrutinizer (Head)** A single-layer **Transformer Encoder** that activates *only* when the stem encounters a high-risk distractor (e.g., a face or complex toy). It leverages self-attention to capture global geometric ratios, rejecting structural false positives that easily fool standard CNNs.

### Optimization: Risk-Weighted Focal Loss (RWFL)
The network is trained using RWFL, a novel objective that maps the HNH40K dataset's explicit taxonomy (High, Moderate, Low risk) into a mathematical penalty parameter. By violently penalizing errors on high-risk objects, the model prioritizes structural hand features over superficial textural similarities.

---

## 📂 Project Structure

```text
HNH40K_Project/
├── HNH40K_Dataset/            # Your dataset root
│   ├── hand/                  # All positive samples
│   └── non-hand/              # All negative samples (distractors)
├── dataset.py                 # Folder-based HNH40K loader 
├── losses.py                  # Risk-Weighted Focal Loss (RWFL)
├── model.py                   # Cascaded Attention Architecture
├── train.py                   # Dual-objective training & evaluation loop
├── main.py                    # Entry point for training execution
├── eval_cm_roc_auc.py         # Exp 1 — Confusion Matrix, ROC/PR curve, AUC, metrics
├── eval_perturbation.py       # Exp 2 — Perturbation robustness analysis
├── eval_new_datasets.py       # Generalization and latency benchmarks on OOD datasets
└── weights/                   # Directory for saved model checkpoints (.pth)
```

---

## 📊 Evaluation & Benchmarking

Three evaluation scripts are provided to comprehensively assess model performance, robustness, and computational efficiency.

### 1. Discrimination Quality (`eval_cm_roc_auc.py`)
This script evaluates the discrimination quality of the trained Cascaded Risk-Aware Filter on a validation split. It generates standard metrics, sweeps classification thresholds, and plots diagnostic curves.

* **Usage**:
  ```bash
  python eval_cm_roc_auc.py --data_dir ./HNH40K_Dataset --checkpoint ./weights/best_cascaded_model.pth --out_dir ./results/exp1_cm_roc_auc
  ```
* **Outputs**:
  - `confusion_matrix.{pdf, png}`: Validation confusion matrix.
  - `roc_curve.{pdf, png}`: ROC and Precision-Recall (PR) curves.
  - `classification_report.txt`: Precision, recall, and F1-score per class.
  - `metrics.{json, csv}`: Overall metrics (accuracy, macro F1, MCC, FPR, FNR).
  - `per_threshold.csv`: Metric sweep across thresholds (0.05 to 0.95).

### 2. Perturbation Robustness (`eval_perturbation.py`)
Analyzes model robustness under common corruption types (ImageNet-C style) across 5 levels of severity. The perturbations include: Gaussian Noise, Gaussian Blur, Brightness, Contrast, JPEG Compression, Occlusion (random patch), and Rotation.

* **Usage**:
  ```bash
  python eval_perturbation.py --data_dir ./HNH40K_Dataset --checkpoint ./weights/best_cascaded_model.pth --out_dir ./results/exp2_perturbation
  ```
* **Outputs**:
  - `perturbation_lineplot.{pdf, png}`: Accuracy curve across severity levels.
  - `perturbation_heatmap.{pdf, png}`: Annotated heatmap showing accuracy per perturbation and severity.
  - `perturbation_bar.{pdf, png}`: Vulnerability profile displaying accuracy drop at maximum severity.
  - `perturbation_table.{csv, json}`: Raw metrics for all tested conditions.

### 3. OOD Generalization & Latency Profiling (`eval_new_datasets.py`)
Evaluates generalization on out-of-distribution (OOD) datasets (e.g., local high-resolution gestures and public low-resolution datasets) while profiling inference latency and early-exit behavior on CPU.

* **Usage**:
  ```bash
  python eval_new_datasets.py
  ```
* **Outputs**:
  - Profiles average latencies on CPU for Stage 1 (Early Exit) vs. the Full Path (Stage 1 + Stage 2).
  - `ood_evaluation_results.json`: Summary stats (samples, recall, early-exit rate, dynamic CPU latency).
  - `ood_evaluation_results.tex`: Copy-pasteable LaTeX tables for the academic paper summarizing OOD performance and latency profiles.
