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
├── data/
│   └── HNH40K_Dataset/        # Your dataset root
│       ├── hand/              # All positive samples
│       └── non-hand/          # All negative samples (distractors)
├── dataset.py                 # Folder-based HNH40K loader 
├── losses.py                  # Risk-Weighted Focal Loss (RWFL)
├── model.py                   # Cascaded Attention Architecture
├── train.py                   # Dual-objective training & evaluation loop
├── main.py                    # Entry point for execution
└── weights/                   # Directory for saved model checkpoints (.pth)