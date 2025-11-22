"""
Evaluate a specific checkpoint on the test set.

Usage:
    python src/eval/evaluate_checkpoint.py --config configs/baseline_resnet50.yaml \
        --checkpoint checkpoints/baseline_resnet50_best_1.pth --device cuda \
        --save-fig --save-preds
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Ensure project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import get_config
from src.models.classifier import create_model
from src.data.dataset import DrowsinessDataset
from src.data.transforms import get_val_transforms
from src.training.metrics import MetricsCalculator


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("epoch", None), ckpt


@torch.no_grad()
def run_eval(model, loader, device="cuda"):
    model.eval()
    metrics_calc = MetricsCalculator(num_classes=2, class_names=["notdrowsy", "drowsy"])
    all_probs = []
    all_labels = []
    for images, labels, metadata in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        metrics_calc.update(preds, labels)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    metrics = metrics_calc.compute()
    metrics["classification_report"] = metrics_calc.get_classification_report()
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return metrics, all_probs, all_labels


def save_confusion_matrix(cm, out_dir: Path):
    import csv
    cm_path = out_dir / "confusion_matrix_test.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_notdrowsy", "pred_drowsy"])
        w.writerow(["actual_notdrowsy", int(cm[0,0]), int(cm[0,1])])
        w.writerow(["actual_drowsy", int(cm[1,0]), int(cm[1,1])])
    return cm_path


def save_figures(cm, probs, labels, roc_auc_scores, out_dir: Path):
    labels_txt = ["notdrowsy", "drowsy"]
    fig1, ax1 = plt.subplots(figsize=(5.2,4.2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels_txt, yticklabels=labels_txt, ax=ax1)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix (Test)")
    f1_path = out_dir / "confusion_matrix_test.png"
    fig1.tight_layout(); fig1.savefig(f1_path, dpi=140); plt.close(fig1)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)
    fig2, ax2 = plt.subplots(figsize=(5.2,4.2))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', cbar=False,
                xticklabels=labels_txt, yticklabels=labels_txt, ax=ax2, vmin=0, vmax=1)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (Normalized)")
    f2_path = out_dir / "confusion_matrix_test_normalized.png"
    fig2.tight_layout(); fig2.savefig(f2_path, dpi=140); plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6.5,5.5))
    colors = ['#1f77b4', '#ff7f0e']
    for class_idx, (cname, color) in enumerate(zip(labels_txt, colors)):
        bin_labels = (labels == class_idx).astype(int)
        class_probs = probs[:, class_idx]
        fpr, tpr, _ = roc_curve(bin_labels, class_probs)
        ax3.plot(fpr, tpr, color=color, lw=2,
                 label=f"{cname} (AUC={roc_auc_scores[cname]:.3f})")
    ax3.plot([0,1],[0,1],"k--", lw=1.2)
    ax3.set_xlim([0,1]); ax3.set_ylim([0,1.02])
    ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curves (Test Set)")
    ax3.legend(loc="lower right"); ax3.grid(alpha=0.3)
    roc_path = out_dir / "roc_curves_test.png"
    fig3.tight_layout(); fig3.savefig(roc_path, dpi=140); plt.close(fig3)

    return f1_path, f2_path, roc_path


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on test set")
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    ap.add_argument("--device", default=None, help="cuda or cpu")
    ap.add_argument("--output-dir", default=None, help="Directory to save metrics (default runs/<exp>_manual_eval)")
    ap.add_argument("--save-fig", action="store_true", help="Save confusion matrix + ROC figures")
    ap.add_argument("--save-preds", action="store_true", help="Save per-sample predictions CSV")
    args = ap.parse_args()

    config = get_config(args.config)
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"
    print(f"Device: {device}")

    exp_name = config.get("logging", {}).get("experiment_name", "experiment")
    base_out = Path(config.get("logging", {}).get("log_dir", "runs"))
    out_dir = Path(args.output_dir) if args.output_dir else (base_out / f"{exp_name}_manual_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = config.get("data", {}).get("image_size", 224)
    val_transform = get_val_transforms(image_size)
    test_csv = config.get("data", {}).get("test_csv")
    data_root = config.get("data", {}).get("data_root", "datasets")
    if test_csv is None:
        raise ValueError("Config missing data.test_csv")

    test_dataset = DrowsinessDataset(
        csv_path=test_csv,
        data_root=data_root,
        transform=val_transform,
        label_map={"notdrowsy": 0, "drowsy": 1},
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("data", {}).get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("data", {}).get("num_workers", 4),
        pin_memory=config.get("data", {}).get("pin_memory", True),
    )
    print(f"Test batches: {len(test_loader)}")

    model = create_model(config).to(device)
    epoch_loaded, ckpt = load_checkpoint(model, Path(args.checkpoint), device)
    print(f"Loaded checkpoint from epoch: {epoch_loaded}")

    print("\nRunning evaluation on test set...")
    metrics, all_probs, all_labels = run_eval(model, test_loader, device=device)

    # ROC AUC per class
    roc_auc_scores = {}
    for idx, cname in enumerate(["notdrowsy", "drowsy"]):
        bin_labels = (all_labels == idx).astype(int)
        class_probs = all_probs[:, idx]
        fpr, tpr, _ = roc_curve(bin_labels, class_probs)
        roc_auc_scores[cname] = auc(fpr, tpr)
    macro_auc = float(np.mean(list(roc_auc_scores.values())))
    roc_auc_scores["macro"] = macro_auc

    # Print summary
    print("\n" + "="*60)
    print("TEST SET EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint Epoch:    {epoch_loaded}")
    print(f"Accuracy:            {metrics['accuracy']:.4f}")
    print(f"Macro Precision:     {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:        {metrics['recall_macro']:.4f}")
    print(f"Macro F1:            {metrics['f1_macro']:.4f}")
    print(f"ROC AUC (macro):     {macro_auc:.4f}")
    print("\nPer-Class ROC AUC:")
    print(f"  notdrowsy:         {roc_auc_scores['notdrowsy']:.4f}")
    print(f"  drowsy:            {roc_auc_scores['drowsy']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save metrics JSON
    metrics_out = out_dir / "metrics_test_manual.json"
    serializable = {}
    for k, v in metrics.items():
        if hasattr(v, "tolist"):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    serializable["roc_auc"] = roc_auc_scores
    serializable["checkpoint_epoch"] = epoch_loaded
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    # Confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    cm_csv = save_confusion_matrix(cm, out_dir)

    # Optional figures
    if args.save_fig:
        f_raw, f_norm, f_roc = save_figures(cm, all_probs, all_labels, roc_auc_scores, out_dir)
        print("\nSaved figures:")
        print(f"  - {f_raw}")
        print(f"  - {f_norm}")
        print(f"  - {f_roc}")

    # Optional predictions CSV
    if args.save_preds:
        import csv
        preds_path = out_dir / "predictions_test_manual.csv"
        # Re-run loop for filenames
        rows = []
        with torch.no_grad():
            for images, labels, metadata in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy()
                labels_np = labels.numpy()
                for i in range(len(labels_np)):
                    rows.append({
                        "filename": metadata["filename"][i],
                        "true": int(labels_np[i]),
                        "pred": int(preds[i]),
                        "prob_notdrowsy": float(probs[i,0]),
                        "prob_drowsy": float(probs[i,1])
                    })
        with open(preds_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["filename","true","pred","prob_notdrowsy","prob_drowsy"])
            w.writeheader(); w.writerows(rows)
        print(f"\nSaved predictions: {preds_path}")

    # Quick summary table
    summary_path = out_dir / "summary_table.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Metric\tValue\n")
        f.write(f"Accuracy\t{metrics['accuracy']:.4f}\n")
        f.write(f"Macro_F1\t{metrics['f1_macro']:.4f}\n")
        f.write(f"Macro_AUC\t{macro_auc:.4f}\n")
        f.write(f"EpochLoaded\t{epoch_loaded}\n")

    print("\n" + "="*60)
    print("SAVED ARTIFACTS")
    print("="*60)
    print(f"  - {metrics_out}")
    print(f"  - {cm_csv}")
    print(f"  - {summary_path}")
    if args.save_fig:
        print(f"  - confusion_matrix_test.png / normalized / roc_curves_test.png")
    if args.save_preds:
        print(f"  - predictions_test_manual.csv")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
