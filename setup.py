"""Setup script for NTHU Driver Drowsiness Detection package"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="nthu-drowsiness-detection",
    version="0.1.0",
    author="NTHU Drowsiness Detection Team",
    description="Image-based driver drowsiness detection on NTHU-DDD2 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmolhem/nthu-driver-drowsiness-ROI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.13.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "albumentations>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
