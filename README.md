# Truth vs. Deception: A Benchmarking Framework for Deepfake Detection Tools

This repository accompanies the Master’s Thesis *“Truth vs. Deception: A Benchmarking Framework for Deepfake Detection Tools”* (Rotterdam School of Management, Erasmus University, 2025). It brings together curated datasets, reproducible training pipelines, and evaluation utilities designed to make head‑to‑head comparisons between state‑of‑the‑art deepfake detectors straightforward and transparent.

---

## Overview

The framework quantifies how different neural‑network architectures perform when faced with diverse forgeries and compression settings. Beyond raw accuracy, it facilitates side‑by‑side inspection of robustness, inference speed, and model interpretability, enabling researchers to pinpoint architectures that best fit their constraints and threat models.

---

## Datasets

| Dataset             | Compression Variants                        | Status in Repo                                                                                       |
| ------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| **CelebDF (v2)**    | High‑quality originals & manipulated videos | ✔ Pre‑processed & split into `train/`, `test/`, `val/`                                               |
| **FaceForensics++** | *C23* & *C40* (H.264 compression levels)    | ✔ Pre‑processed splits `train/`, `test/`, `val/`

All videos have been converted to aligned face crops at a uniform resolution so that training scripts can work plug‑and‑play across sources.

---

## Repository Layout

```text
├── datasets/
│   ├── celeb_df_v2/
│   │   ├── train/
│   │   ├── test/
│   │   └── val/
│   └── faceforensics++/       # upcoming C23 & C40 splits
│
├── scripts/
│   ├── data_splitting/
│   │   └── split_and_structure_real_fake.py
│   ├── frame_extraction/
│   │   └── video_face_extractor.py
│   ├── training_models/
│   │   ├── train_hybrid.py
│   │   ├── train_mobilenetv3.py
│   │   ├── train_vit.py
│   │   └── train_xception.py
│   ├── evaluate_hybrid.py
│   ├── evaluate_mobilenetv3.py
│   ├── evaluate_vit_b16.py
│   └── evaluate_xception.py
│
├── src/                        # supporting utility modules
├── install_dependencies.py     # one‑shot environment setup helper
└── README.md
```

* **datasets/** – Drop‑in location for the aligned face crops used in experiments.
* **scripts/** – Modular helpers for splitting raw data, extracting frames, launching training runs, and generating evaluation metrics.

---

## Detection Architectures Benchmarked

| Category                 | Implementations in `scripts/training_models/` |
| ------------------------ | --------------------------------------------- |
| **CNN Backbone**         | `train_xception.py` (Xception)                |
| **Vision Transformer**   | `train_vit.py` (ViT‑B/16)                     |
| **Hybrid / Ensemble**    | `train_hybrid.py` (CNN + Transformer fusion)  |
| **Lightweight / Mobile** | `train_mobilenetv3.py` (MobileNet V3)         |

Each training script saves checkpoints, TensorBoard logs, and automatically triggers the paired *evaluate* script to generate ROC curves and class‑balanced metrics.

---

## Getting Started at a Glance

1. **Prepare data** – Place the pre‑processed folders shown above inside `datasets/` (or wait for the upcoming FaceForensics++ drops).
2. **Launch an experiment** – Pick a training script from `scripts/training_models/` and run it; the framework handles loader selection, augmentation, logging, and evaluation out of the box.
3. **Inspect results** – Review the generated metrics and visualisations to compare models on your chosen dataset splits.

---

## Contributing

Bug reports, pull requests, and dataset extension suggestions are warmly welcomed. Opening an issue with a short reproducible example is the fastest way to get something addressed.

---

## Author

**Leon Tubbing** – MSc Business Information Management, RSM, Erasmus University, 2025.

## Supervisor

**Anna Priante** – Rotterdam School of Management, Erasmus University.
