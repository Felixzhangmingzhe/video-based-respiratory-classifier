# Respiratory Pattern Classification from ICU Chest Videos (CSC494/CSC495)

This repository contains the code, configuration files, and reports for my CSC494 and CSC495 research project on classifying respiratory patterns from ICU chest videos using optical flow, frequency-domain features, and a multi-layer perceptron (MLP) classifier.

> **Courses:** CSC494/CSC495: Supervised Computer Science Research Project  
> **Institution:** University of Toronto  
> **Student:** Mingzhe Zhang  
> **Supervisor:** Babak Taati

---

## 1. Project Overview

The goal of this project is to detect and classify **breathing events** from ICU thorax videos into:
- `normal`
- `abnormal`
- `unsure`

The pipeline is:

1. **Annotation**  
   - Each breath is annotated with `start_time`, `end_time`, and a label (`normal`, `abnormal`, `unsure`).
   - Annotations are stored in CSV files using a shared time base with the video.

2. **Signal Extraction (Video → flow.npy / volume.npy)**  
   - A thorax region of interest (ROI) is defined for each video.
   - Optical flow is computed on sparse keypoints (Shi–Tomasi + Lucas–Kanade).
   - The magnitude of motion is averaged over points to obtain a 1D **flow** signal (z-score normalized).
   - The **volume** signal is obtained by integrating the flow over time.

3. **Windowing and Labeling**  
   - The signals are segmented into sliding windows of `N = 256` frames (≈ 8.53 s at 30 fps), stride `1` frame.
   - A window is labeled **normal** only if all complete breaths inside are `normal`; otherwise, it is treated as non-normal (e.g., `abnormal` or `unsure` depending on the experiment).

4. **Feature Extraction (FFT)**  
   - For each window, frequency-domain features are computed using the FFT of the flow or volume signal.

5. **Classification with MLP**  
   - An MLP is trained on the FFT features to classify the window label.
   - Experiments include per-video models (CSC494) and a single cross-video model with cross-validation (CSC495).

---

## 2. CSC494 vs CSC495

Although CSC494 and CSC495 are part of the same research project, they are evaluated separately:

### CSC494 (Proof-of-Concept)
- **Scope:** per-video proof-of-concept.
- Each video is processed separately.
- For each video, a separate MLP is trained.
- Reports **per-video** performance (accuracy, confusion matrix, precision/recall, ROC/AUC).

### CSC495 (Scaling Up)
- **Scope:** multi-video training and evaluation.
- All windows from all videos are combined.
- A **single MLP** is trained with **5-fold stratified cross-validation**.
- Reports cross-validation metrics, and optionally performance broken down by video.

The same core pipeline (annotation → signal extraction → windowing → FFT → MLP) is used in both parts.

---

## 3. Repository Structure

```text
.
├── README.md
├── requirements.txt          # Python dependencies
├── configs/                  # Experiment configuration files
│   ├── csc494_video1.yaml
│   ├── csc494_video2.yaml
│   └── csc495_crossval.yaml
├── data/
│   ├── raw/                  # ICU videos (NOT included in this repo)
│   ├── processed/            # flow.npy, volume.npy (gitignored)
│   └── annotations/          # CSV annotation files (possibly anonymized)
├── src/
│   ├── preprocessing/        # Video → flow/volume
│   │   ├── extract_flow.py
│   │   └── make_volume.py
│   ├── features/             # Windowing + FFT features
│   │   └── window_fft.py
│   ├── models/               # MLP model and training utilities
│   │   ├── mlp.py
│   │   └── train_utils.py
│   └── experiments/          # Entry points for running experiments
│       ├── run_csc494_video1.py
│       ├── run_csc494_video2.py
│       └── run_csc495_crossval.py
├── notebooks/                # Jupyter notebooks for exploration/debugging
│   ├── 01_explore_signals.ipynb
│   └── 02_debug_model.ipynb
└── reports/                  # Project reports (PDF, LaTeX, slides, etc.)
    ├── csc494_report/
    └── csc495_report/
