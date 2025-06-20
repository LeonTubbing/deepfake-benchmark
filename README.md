```markdown
# Truth vs. Deception: A Benchmarking Framework for Deepfake Detection Tools

This repository contains a comprehensive benchmarking framework developed as part of a Master's Thesis titled "Truth vs. Deception: A Benchmarking Framework for Deepfake Detection Tools" at the Rotterdam School of Management (RSM), Erasmus University, 2025.

## Overview

The purpose of this project is to systematically compare the performance of various deepfake detection tools using standardized conditions and datasets. The benchmarking evaluates several deep learning architectures, focusing on accuracy, robustness, interpretability, and computational efficiency.

## Datasets

Two publicly available deepfake datasets have been used:

- **FaceForensics++**: Includes original and manipulated videos created using various deepfake methods (Deepfakes, FaceSwap, Face2Face, etc.) with different compression levels.  
- **CelebDF (v2)**: Contains high-quality deepfake videos of celebrities, featuring natural and refined manipulations.

The datasets have been preprocessed uniformly (frames extracted, faces cropped and aligned, resolution standardized) to ensure fair comparisons.

## Repository Structure

```

.
├── datasets/                             # Preprocessed datasets go here
├── install\_dependencies.py              # Installs all Python requirements
└── scripts/
├── split\_and\_structure\_real\_fake\_dataset.py
├── video\_face\_extractor.py
├── train\_xception.py
├── train\_vit.py
├── train\_mobilenetv3.py
├── train\_hybrid.py
├── evaluate\_xception.py
├── evaluate\_vit\_b16.py
├── evaluate\_mobilenetv3.py
└── evaluate\_hybrid.py

````

## install_dependencies.py

```python
#!/usr/bin/env python3
"""
install_dependencies.py

Install all required Python packages for this deepfake‑detection benchmarking framework.
Run with:
    python install_dependencies.py
"""

import subprocess
import sys

def install_packages(packages):
    """Run pip install on the given list of packages."""
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    subprocess.check_call(cmd)

if __name__ == "__main__":
    packages = [
        "tensorflow>=2.9.0",
        "vit-keras",
        "scikit-learn",
        "matplotlib",
        "opencv-python",
        "mtcnn",
        "tqdm",
        "numpy"
    ]
    print("Installing dependencies:")
    for pkg in packages:
        print("  -", pkg)
    install_packages(packages)
    print("\n✅ All dependencies installed successfully.")
````

## Detection Tools Evaluated

* **CNN-Based Models** (Xception)
* **Vision Transformer-Based Models** (ViT)
* **Hybrid Architectures** (Ensemble and multi-stream models combining CNN and Transformers)
* **Lightweight Models** (MobileNet v3)

## Quickstart

1. **Clone the repo**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install dependencies**

   ```bash
   python install_dependencies.py
   ```

3. **Prepare your datasets**
   Place (or download) your preprocessed FaceForensics++ and CelebDF data under:

   ```
   datasets/
   ```

4. **Preprocess & split**

   ```bash
   python scripts/split_and_structure_real_fake_dataset.py
   python scripts/video_face_extractor.py
   ```

5. **Train a model**
   For example, to train Xception:

   ```bash
   python scripts/train_xception.py
   ```

   Or any of:

   ```bash
   python scripts/train_vit.py
   python scripts/train_mobilenetv3.py
   python scripts/train_hybrid.py
   ```

6. **Evaluate performance**

   ```bash
   python scripts/evaluate_xception.py
   python scripts/evaluate_vit_b16.py
   python scripts/evaluate_mobilenetv3.py
   python scripts/evaluate_hybrid.py
   ```

## Contributions and Further Information

This repository is intended for researchers and developers in deepfake detection. Contributions, feedback, and suggestions are welcome—please open an issue or pull request on GitHub.

For detailed methods, experiments, and results, see the accompanying Master’s Thesis.

## Author

**Leon Tubbing**
MSc Business Information Management, RSM, Erasmus University, 2025

## Supervisor

**Anna Priante**
Rotterdam School of Management, Erasmus University

```
```
