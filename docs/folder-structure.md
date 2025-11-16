# Project Folder Structure

## Overview

This document describes the folder structure of the NTHU Driver Drowsiness Detection with ROI-based approach project.

```text
nthu-driver-drowsiness-ROI/
├── .venv/                          # Python virtual environment (ignored by git)
├── .vscode/                        # VS Code workspace settings
│   ├── settings.json              # Python interpreter and workspace config
│   └── extensions.json            # Recommended extensions
├── data/                          # Processed data and manifests
│   └── manifests/
│       └── archive_manifest.csv   # Dataset manifest (66,521 images)
├── datasets/                      # Raw datasets (ignored by git)
│   └── archive/
│       ├── drowsy/                # Drowsy-labeled images (36,030 files)
│       └── notdrowsy/             # Not-drowsy labeled images (30,491 files)
├── docs/                          # Project documentation
│   ├── env-setup.md              # Environment setup instructions
│   ├── folder-structure.md       # This file
│   └── proposal.md               # Full project proposal
├── src/                           # Source code
│   ├── __init__.py               # Package initialization
│   └── data/
│       └── build_archive_manifest.py  # Script to generate dataset manifest
├── .gitignore                     # Git ignore rules
├── .python-version                # Python version specification
├── README.md                      # Project overview
└── requirements.txt               # Python dependencies
```

## Directory Descriptions

### Core Directories

#### `data/`

Contains processed data, manifests, and metadata.

- **manifests/**: CSV files mapping image paths to metadata (subject, behavior, glasses status, label)

#### `datasets/`

Raw dataset storage (excluded from version control).

- **archive/**: NTHU-DDD dataset with drowsy/notdrowsy classification
  - Images follow naming convention: `{subject}_{glasses}_{behavior}_{frame}_{label}.jpg`

#### `docs/`

Project documentation including setup guides, proposals, and architecture docs.

#### `src/`

Source code organized by functionality:

- **data/**: Data loading, preprocessing, and manifest building scripts

### Configuration Files

- **`.venv/`**: Python 3.10.11 virtual environment with TensorFlow, Keras, OpenCV
- **`.vscode/`**: VS Code settings for automatic virtual environment activation
- **`.gitignore`**: Excludes datasets, virtual environment, and generated artifacts
- **`requirements.txt`**: Pinned Python package versions

## Dataset Statistics

- **Total Images**: 66,521
- **Subjects**: 4 (001, 002, 005, 006)
- **Labels**: drowsy (36,030), notdrowsy (30,491)
- **Behaviors**: nonsleepyCombination, sleepyCombination, slowBlinkWithNodding, yawning
- **Variants**: glasses/noglasses

## Filename Convention

Dataset images follow this pattern:

```text
{subject_id}_{glasses_status}_{behavior}_{frame_number}_{label}.jpg
```

**Example:**

```text
006_glasses_sleepyCombination_1868_drowsy.jpg
```

Where:

- `subject_id`: 001, 002, 005, 006
- `glasses_status`: glasses, noglasses
- `behavior`: sleepyCombination, nonsleepyCombination, slowBlinkWithNodding, yawning
- `frame_number`: Sequential frame index from video
- `label`: drowsy, notdrowsy

## Generated Artifacts (Not in Git)

The following directories will be created during development but are excluded from version control:

- **`runs/`**: Training run logs, metrics, and checkpoints
- **`reports/`**: Generated figures, tables, and analysis results
- **`notebooks/`**: Jupyter notebooks for exploration and experiments
- **`checkpoints/`**: Model checkpoints during training
- **`artifacts/`**: Exported models and predictions
- **`models/`**: Saved trained models

## Development Workflow

1. **Environment Setup**: Activate `.venv` (automatically done by VS Code)
2. **Data Preparation**: Use `src/data/build_archive_manifest.py` to generate manifests
3. **Development**: Add new modules under `src/` organized by functionality
4. **Documentation**: Update relevant docs in `docs/` as needed
5. **Version Control**: Only commit code and docs, not datasets or artifacts

## Notes

- All paths in scripts use relative references from project root
- Dataset files remain in their original location (`datasets/archive/`)
- Manifest CSVs contain relative paths to images for portability
- Python 3.10.11 required with dependencies in `requirements.txt`

