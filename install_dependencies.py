"""
install_dependencies.py

Install all required Python packages for this deepfake‑detection repository.
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
