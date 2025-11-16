# Dataset Manifest Documentation

## Overview

The dataset manifest (`data/manifests/archive_manifest.csv`) is a comprehensive index of all images in the NTHU Driver Drowsiness Detection dataset. It provides structured metadata for each image, enabling efficient data loading, filtering, and subject-exclusive splitting.

## Manifest Structure

### File Location

```text
data/manifests/archive_manifest.csv
```

### CSV Schema

The manifest contains the following columns:

| Column     | Type   | Description                                              | Example Values                                  |
|------------|--------|----------------------------------------------------------|-------------------------------------------------|
| `filename` | string | Relative path to image from `datasets/archive/`          | `drowsy/006_glasses_sleepyCombination_1868_drowsy.jpg` |
| `label`    | string | Drowsiness classification                                | `drowsy`, `notdrowsy`                           |
| `subject`  | string | Subject ID (zero-padded)                                 | `001`, `002`, `005`, `006`                      |
| `glasses`  | string | Whether subject is wearing glasses                       | `glasses`, `noglasses`                          |
| `behavior` | string | Behavioral scenario during recording                     | `sleepyCombination`, `nonsleepyCombination`, `slowBlinkWithNodding`, `yawning` |
| `frame`    | string | Frame number extracted from video sequence               | `0`, `1`, `100`, `1868`                         |

### Sample Records

```csv
filename,label,subject,glasses,behavior,frame
drowsy/006_glasses_sleepyCombination_1868_drowsy.jpg,drowsy,006,glasses,sleepyCombination,1868
notdrowsy/001_glasses_nonsleepyCombination_36_notdrowsy.jpg,notdrowsy,001,glasses,nonsleepyCombination,36
drowsy/005_noglasses_slowBlinkWithNodding_1793_drowsy.jpg,drowsy,005,noglasses,slowBlinkWithNodding,1793
```

## Dataset Statistics

### Overall Summary

- **Total Images**: 66,521
- **Total Subjects**: 4
- **Behaviors**: 4 distinct scenarios
- **Glasses Variants**: With/without glasses

### Label Distribution

| Label       | Count  | Percentage |
|-------------|--------|------------|
| drowsy      | 36,030 | 54.2%      |
| notdrowsy   | 30,491 | 45.8%      |

**Class Balance**: Slightly imbalanced with more drowsy samples. Consider using weighted loss or stratified sampling during training.

### Subject Distribution

| Subject ID | Image Count | Percentage |
|------------|-------------|------------|
| 001        | 19,016      | 28.6%      |
| 002        | 18,833      | 28.3%      |
| 005        | 21,933      | 33.0%      |
| 006        | 6,739       | 10.1%      |

**Note**: Subject 006 has significantly fewer samples. Ensure proper stratification when creating train/val/test splits to maintain representation.

### Glasses Distribution

| Variant    | Count  | Percentage |
|------------|--------|------------|
| glasses    | 37,050 | 55.7%      |
| noglasses  | 29,471 | 44.3%      |

### Behavior Distribution

| Behavior                | Count  | Percentage | Description                           |
|-------------------------|--------|------------|---------------------------------------|
| nonsleepyCombination    | 20,918 | 31.4%      | Normal alert driving behavior         |
| sleepyCombination       | 19,958 | 30.0%      | Multiple drowsiness indicators        |
| slowBlinkWithNodding    | 13,147 | 19.8%      | Slow eye blinks with head nodding     |
| yawning                 | 12,498 | 18.8%      | Yawning behavior                      |

## Usage Examples

### Loading Manifest in Python

```python
import pandas as pd

# Load manifest
df = pd.read_csv('data/manifests/archive_manifest.csv')

# Filter by label
drowsy_samples = df[df['label'] == 'drowsy']
notdrowsy_samples = df[df['label'] == 'notdrowsy']

# Filter by subject
subject_001 = df[df['subject'] == '001']

# Filter by behavior
yawning_samples = df[df['behavior'] == 'yawning']

# Filter by glasses status
glasses_samples = df[df['glasses'] == 'glasses']

# Get full image path
from pathlib import Path
base_path = Path('datasets/archive')
image_paths = df['filename'].apply(lambda x: base_path / x)
```

### Subject-Exclusive Splitting

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load manifest
df = pd.read_csv('data/manifests/archive_manifest.csv')

# Get unique subjects
subjects = df['subject'].unique()

# Split subjects (not samples)
train_subjects, test_subjects = train_test_split(
    subjects, test_size=0.2, random_state=42
)
train_subjects, val_subjects = train_test_split(
    train_subjects, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
)

# Create subject-exclusive splits
train_df = df[df['subject'].isin(train_subjects)]
val_df = df[df['subject'].isin(val_subjects)]
test_df = df[df['subject'].isin(test_subjects)]

print(f"Train: {len(train_df)} samples, {len(train_subjects)} subjects")
print(f"Val: {len(val_df)} samples, {len(val_subjects)} subjects")
print(f"Test: {len(test_df)} samples, {len(test_subjects)} subjects")
```

### Filtering by Multiple Criteria

```python
import pandas as pd

df = pd.read_csv('data/manifests/archive_manifest.csv')

# Get drowsy samples from subject 005 wearing glasses
filtered = df[
    (df['label'] == 'drowsy') &
    (df['subject'] == '005') &
    (df['glasses'] == 'glasses')
]

# Get all yawning or slowBlinkWithNodding behaviors
drowsy_behaviors = df[
    df['behavior'].isin(['yawning', 'slowBlinkWithNodding'])
]
```

## Manifest Generation

The manifest is generated using the `src/data/build_archive_manifest.py` script.

### Regenerating the Manifest

```bash
# Activate virtual environment (done automatically in VS Code)
.venv\Scripts\activate

# Run manifest builder
python src/data/build_archive_manifest.py
```

### Builder Script Features

- **Automatic Discovery**: Recursively scans `datasets/archive/drowsy/` and `datasets/archive/notdrowsy/`
- **Filename Parsing**: Extracts metadata from standardized filename format
- **Validation**: Skips malformed filenames with error reporting
- **Sorting**: Orders records by subject, behavior, and frame for consistency
- **Statistics**: Prints distribution summaries after generation

## Data Quality Considerations

### Filename Format Validation

All images must follow this naming convention:

```text
{subject}_{glasses}_{behavior}_{frame}_{label}.jpg
```

Images not matching this pattern will be skipped during manifest generation.

### Missing Data

- No missing values in any columns
- All records have complete metadata
- Frame numbers are preserved as strings to maintain leading zeros if present

### Temporal Information

The `frame` column indicates the sequential frame number from the original video recording. This can be useful for:

- **Temporal analysis**: Understanding drowsiness onset patterns
- **Sequential sampling**: Creating video-like sequences for RNNs/LSTMs
- **Data leakage prevention**: Avoiding temporal leakage by splitting on subjects, not frames

## Best Practices

### For Training

1. **Subject-Exclusive Splits**: Always split by subject, never by individual frames
2. **Stratification**: Maintain label balance across splits
3. **Subject 006**: Monitor performance on this underrepresented subject
4. **Reproducibility**: Use fixed random seeds when splitting

### For Evaluation

1. **Report per-subject metrics**: Understand model generalization to new subjects
2. **Report per-behavior metrics**: Identify which behaviors are harder to classify
3. **Glasses robustness**: Evaluate performance separately for glasses/noglasses
4. **Class balance**: Use macro-averaged metrics (macro-F1) to account for imbalance

### For Data Loading

1. **Relative paths**: Manifest uses relative paths from `datasets/archive/`
2. **Path construction**: Join manifest paths with base dataset directory
3. **Existence checks**: Verify images exist before training
4. **Caching**: Consider caching loaded images for faster iteration

## Related Files

- **Manifest CSV**: `data/manifests/archive_manifest.csv`
- **Builder Script**: `src/data/build_archive_manifest.py`
- **Dataset Directory**: `datasets/archive/`
- **Project Structure**: See `docs/folder-structure.md`

## Updates and Maintenance

### When to Regenerate

Regenerate the manifest if:

- New images are added to the dataset
- Images are removed or renamed
- Dataset directory structure changes
- Metadata extraction logic is updated

### Version Control

- The manifest CSV is **tracked in Git** (not in `.gitignore`)
- Builder script is **tracked in Git**
- Actual dataset images are **not tracked** (excluded in `.gitignore`)

This allows the manifest to be shared with collaborators without transferring the large dataset files.

## Future Enhancements

Potential additions to the manifest:

- **Split column**: Pre-assigned train/val/test splits
- **Image dimensions**: Width and height for each image
- **Face detection confidence**: Metadata from preprocessing
- **ROI coordinates**: Bounding boxes for eyes, mouth, face regions
- **Quality metrics**: Blur detection, lighting conditions, occlusion flags
- **Augmentation flags**: Mark augmented samples vs. original data

