# NTHU Driver Drowsiness Detection with ROI Priors

Image-based driver drowsiness detection on the NTHU-DDD2 dataset using single-frame CNNs and region-of-interest (ROI) priors.  
We focus on:
- **Subject-exclusive splits** (no identity leakage)
- **Eye/mouth ROI masks** as priors
- **Compact backbones** vs heavier CNNs
- **Robustness** to glare, blur, and eyelid occlusion

---

## Dataset

We use a subset of the NTHU-DDD2 drowsiness dataset:

- Scenario: **day + glasses**
- Subjects: 4 drivers
- Classes: `normal`, `slow_blink`, `yawn`, `sleepy`

Download the dataset and place it under:

```text
datasets/
  nthu_ddd2_raw/          # original download
  nthu_ddd2_day_glasses/  # filtered subset (4 subjects, day+glasses)
