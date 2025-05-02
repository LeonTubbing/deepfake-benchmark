# Truth vs. Deception: A Benchmarking Framework for Deepfake Detection Tools

This repository contains a comprehensive benchmarking framework developed as part of a Master's Thesis titled "Truth vs. Deception: A Benchmarking Framework for Deepfake Detection Tools" at the Rotterdam School of Management (RSM), Erasmus University, 2025.

## Overview

The purpose of this project is to systematically compare the performance of various deepfake detection tools using standardized conditions and datasets. The benchmarking evaluates several deep learning architectures, focusing on accuracy, robustness, interpretability, and computational efficiency.

## Datasets

Two publicly available deepfake datasets have been used:

* **FaceForensics++**: Includes original and manipulated videos created using various deepfake methods (Deepfakes, FaceSwap, Face2Face, etc.) with different compression levels.
* **CelebDF (v2)**: Contains high-quality deepfake videos of celebrities, featuring natural and refined manipulations.

The datasets have been preprocessed uniformly (frames extracted, faces cropped and aligned, resolution standardized) to ensure fair comparisons.

## Repository Structure

* **datasets/**: Location for storing and accessing preprocessed datasets.
* **scripts/**: Contains Python scripts for:

  * Data preprocessing (extracting and aligning frames)
  * Training, validation, and testing of deepfake detection models
  * Evaluation and visualization of results

## Detection Tools Evaluated

The following deepfake detection architectures have been benchmarked:

* **CNN-Based Models** (e.g., Xception, MesoNet)
* **Vision Transformer-Based Models** (ViT)
* **Hybrid Architectures** (Ensemble and multi-stream models combining CNN and Transformers)
* **Lightweight Models** (MobileNet variants)

## How to Use

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Place or download preprocessed datasets in the `datasets/` folder.

4. Use provided scripts in `scripts/` folder to preprocess data, train models, and evaluate performance:

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
```

## Contributions and Further Information

This repository is primarily intended for researchers and developers interested in deepfake detection. Contributions, feedback, and suggestions are welcome. Please contact the repository owner or submit issues via GitHub.

For a detailed explanation of methods, experiments, and results, please refer to the accompanying Master's Thesis document.

## Author

* **Leon Tubbing** – MSc Business Information Management, RSM, Erasmus University, 2025.

## Supervisor

* **Anna Priante** – RSM, Erasmus University.
