# HNH40K: A Robust Dataset and Risk-Weighted Learning for Hand Filtering

Ashikuzzaman†, Md Selim Sarowar†, Md Tanvir Islam, Sangtae Ahn, and Khan Muhammad

Accepted at the British Machine Vision Conference (BMVC 2026). [OpenReview](https://openreview.net/forum?id=C23KZbQblz)

PyTorch code for the **Cascaded Risk-Aware Filter**, a two-stage hand vs. non-hand classifier trained with a **Risk-Weighted Focal Loss (RWFL)** on the HNH40K dataset.

## Overview

The filter is a front-end gate: it decides whether a frame contains a hand before a heavier model runs. Edge CNNs confuse hands with look-alike distractors (faces, printed hands, toys), and global-context models cost too much for early filtering. The model splits the work in two stages:

1. **Stage 1, fast rejector.** The first 7 feature blocks of an ImageNet-pretrained `MobileNetV3-Small` (40 output channels), followed by global average pooling and a linear early-exit classifier.
2. **Stage 2, structural scrutinizer.** A single Transformer encoder layer (4 heads, feed-forward width 128) over the flattened stem feature map, mean pooling, and an MLP classifier (40 to 64 to 1).

At inference with batch size 1, the model returns the Stage 1 logit and skips Stage 2 when the early probability is below the exit threshold (0.1). With larger batches, Stage 2 always runs.

**Loss.** RWFL multiplies the binary focal loss (alpha 0.25, gamma 2.0) by a per-sample risk weight. Training sums the Stage 2 loss and 0.4 times the Stage 1 loss.

**Current state of the risk weights.** `dataset.py` assigns a risk weight of 1.0 to every sample, so the committed training code runs RWFL as a plain focal loss. To train with the risk taxonomy described in [ARCHITECTURE.md](ARCHITECTURE.md) (2.0 for high-risk negatives, 0.5 for low-risk backgrounds), set per-category weights in `dataset.py`.

[ARCHITECTURE.md](ARCHITECTURE.md) has the full design notes and an inference flow diagram.

## Repository layout

```text
RWFL-HNH40K/
├── model.py                    # CascadedRiskAwareFilter (stem, early exit, attention head)
├── losses.py                   # RiskWeightedFocalLoss
├── dataset.py                  # HNH40KBinaryDataset: folder loader for hand/ and non-hand/
├── train.py                    # dual-objective training loop and per-epoch validation
├── main.py                     # entry point: --mode train | evaluate
├── eval_cm_roc_auc.py          # Exp 1: confusion matrix, ROC and PR curves, threshold sweep
├── eval_perturbation.py        # Exp 2: robustness to 7 perturbations at 5 severities
├── eval_new_datasets.py        # out-of-distribution recall and CPU latency
├── compute_table_results.py    # metrics at 5K, 10K, 20K, 40K subsets; model profile
├── generate_plots.py           # confusion matrix PDF for the validation split
├── check_data.py               # prints hand and non-hand image counts
├── ARCHITECTURE.md             # design notes
├── HandGestureDataset/         # 18,000 JPEGs in 20 class folders (0 to 19), used by eval_new_datasets.py
├── 02.Supp (1).tex             # supplementary material source (review copy)
├── bmvc2k.cls, subfigure.sty   # LaTeX class and style for the supplement
└── 02.Supp (1).log             # LaTeX build log
```

The HNH40K images and trained weights are not in this repository (`HNH40K_Dataset/`, `weights/` and `results/` are git-ignored).

## Installation

There is no requirements file and no pinned Python version. The scripts import PyTorch, torchvision, NumPy, Pillow, Matplotlib, seaborn and scikit-learn:

```bash
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn
```

`model.py` downloads the ImageNet weights for MobileNetV3-Small through torchvision on first use.

## Data

Place the dataset in a folder with two subfolders. Images may sit in nested folders (`.jpg`, `.jpeg`, `.png`, `.bmp`):

```text
HNH40K_Dataset/
├── hand/        # positive samples
└── non-hand/    # negative samples (distractors)
```

All inputs are resized to 192 x 160 and normalized with the ImageNet mean and standard deviation.

## Usage

Most scripts default to paths on the authors' server (`/data/selim_sarowar/HNH40K/...`). Pass your own paths as shown.

### Train

```bash
python main.py --mode train --data_dir ./HNH40K_Dataset --epochs 50 --batch_size 64 --checkpoint ./weights/best_cascaded_model.pth
```

`main.py` splits the dataset 80/20 into training and validation subsets. It uses AdamW (learning rate 1e-4, weight decay 1e-4), cosine annealing and gradient clipping at norm 1.0. The checkpoint with the highest validation accuracy is saved.

### Evaluate

```bash
python main.py --mode evaluate --data_dir ./HNH40K_Dataset --checkpoint ./weights/best_cascaded_model.pth
```

Prints accuracy, false positive rate and the confusion counts at threshold 0.5.

**Split note.** `main.py` does not seed its random split, while `eval_cm_roc_auc.py`, `eval_perturbation.py` and `generate_plots.py` split with seed 42. A validation subset drawn by these scripts is therefore not guaranteed to be disjoint from the training subset of a `main.py` run. Seed the split in `main.py` the same way if you need a held-out evaluation.

### Experiment 1: discrimination quality

```bash
python eval_cm_roc_auc.py --data_dir ./HNH40K_Dataset --checkpoint ./weights/best_cascaded_model.pth --out_dir ./results/exp1_cm_roc_auc
```

Outputs in `--out_dir`:

- `confusion_matrix.{pdf,png}`: validation confusion matrix
- `roc_curve.{pdf,png}`: ROC and precision-recall curves
- `classification_report.txt`: precision, recall and F1 per class
- `metrics.{json,csv}`: accuracy, macro F1, MCC, FPR, FNR and confusion counts
- `per_threshold.csv`: metrics at 19 thresholds from 0.05 to 0.95

### Experiment 2: perturbation robustness

```bash
python eval_perturbation.py --data_dir ./HNH40K_Dataset --checkpoint ./weights/best_cascaded_model.pth --out_dir ./results/exp2_perturbation
```

Applies Gaussian noise, Gaussian blur, brightness, contrast, JPEG compression, occlusion (random patch) and rotation, each at 5 severity levels. Outputs in `--out_dir`:

- `perturbation_lineplot.{pdf,png}`: accuracy against severity
- `perturbation_heatmap.{pdf,png}`: accuracy per perturbation and severity
- `perturbation_bar.{pdf,png}`: accuracy drop at severity 5
- `perturbation_table.{csv,json}`: all metrics for every condition

### Out-of-distribution recall and CPU latency

```bash
python eval_new_datasets.py
```

This script has no command-line arguments. Before running it, edit the four paths at the top of `main()` in `eval_new_datasets.py` (`checkpoint_path`, `local_dataset_dir`, `public_dataset_dir`, `results_dir`). `public_dataset_dir` can point to the `HandGestureDataset/` folder in this repository. The local high-resolution gesture set is not included.

It times the Stage 1 path and the full path on CPU (500 runs after 100 warmup runs, batch size 1), then reports recall and early-exit rate on both datasets. It writes `ood_evaluation_results.json` and `ood_evaluation_results.tex` (LaTeX tables) to `results_dir`.

### Other scripts

```bash
python compute_table_results.py --data_dir ./HNH40K_Dataset --checkpoint ./weights/best_cascaded_model.pth --out_dir ./results
```

Writes `table_metrics.{csv,json}` (metrics on stratified subsets of 5,000, 10,000, 20,000 and 40,000 images) and `table_profiles.{csv,json}` (parameters, FLOPs, size and latency at 128 x 128 and 256 x 256 inputs).

`generate_plots.py` and `check_data.py` take no arguments and use hard-coded server paths. Edit the paths before running them.

## Results

All numbers below are copied from the supplementary material source in this repository, [`02.Supp (1).tex`](<02.Supp (1).tex>). No results files or checkpoints are committed.

**Classification report on the HNH40K 40K validation split** (8,090 samples, seed 42, threshold 0.5). MCC is 0.9990.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-Hand | 0.9995 | 0.9995 | 0.9995 | 3,993 |
| Hand | 0.9995 | 0.9995 | 0.9995 | 4,097 |
| Macro avg | 0.9995 | 0.9995 | 0.9995 | 8,090 |

**Accuracy under perturbation**, same split (clean accuracy 0.9995).

| Perturbation | Severity 1 | Severity 3 | Severity 5 |
|---|---|---|---|
| Gaussian noise | 0.866 | 0.514 | 0.515 |
| Gaussian blur | 0.979 | 0.783 | 0.736 |
| JPEG compression | 0.993 | 0.920 | 0.631 |
| Rotation | 0.992 | 0.905 | 0.861 |
| Brightness | 0.999 | 0.996 | 0.952 |
| Contrast | 0.999 | 0.993 | 0.950 |
| Occlusion | 0.999 | 0.995 | 0.953 |

The model holds up under brightness, contrast and occlusion changes and degrades sharply under Gaussian noise and strong JPEG compression. The supplement lists F1, precision and recall for all five severities.

**Out-of-distribution hand gesture sets** (all images are hands, so the table reports recall).

| Dataset | Source resolution | Samples | Recall (threshold 0.5) | Early-exit rate | Avg. CPU latency |
|---|---|---|---|---|---|
| Local gesture images | 3072 x 3072 | 934 | 86.08% | 3.53% | 9.20 ms |
| [HandGestureDataset](https://huggingface.co/datasets/S1223/HandGestureDataset) | 50 x 50 | 18,000 | 98.49% | 0.00% | 9.21 ms |

**CPU latency per sample** (batch size 1, 500 runs after 100 warmup runs; the supplement does not name the CPU).

| Inference path | Latency | Throughput |
|---|---|---|
| Stage 1 only (early exit) | 8.96 ± 4.09 ms | 111.6 FPS |
| Full path (Stage 1 + Stage 2) | 9.21 ± 2.33 ms | 108.6 FPS |

## Citation

```bibtex
@inproceedings{Ashikuzzaman_2026_BMVC,
author    = {Ashikuzzaman and Md Selim Sarowar and Md Tanvir Islam and Sangtae Ahn and Khan Muhammad},
title     = {HNH40K: A Robust Dataset and Risk-Weighted Learning for Hand Filtering},
booktitle = {37th British Machine Vision Conference 2026, {BMVC} 2026, Lancaster, UK, November 23-26, 2026},
publisher = {BMVA},
year      = {2026},
url       = {https://bmva-archive.org.uk/bmvc/2026/assets/papers/Paper_652/paper.pdf}
}
```

## License

The authors release this repository under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). `HandGestureDataset/` is a copy of a public third-party dataset ([source](https://huggingface.co/datasets/S1223/HandGestureDataset)) and remains under its own terms.
