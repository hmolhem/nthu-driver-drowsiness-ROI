"""
Evaluate a trained drowsiness detection model on the test set.

Usage (Windows PowerShell):
  python src/eval/evaluate_model.py --config configs/baseline_resnet50.yaml \
    --checkpoint checkpoints/baseline_resnet50_best.pth --device cuda

Outputs:
  - Prints overall metrics, per-class metrics, and confusion matrix
  - Saves JSON metrics to runs/<experiment>/metrics_test.json
  - Saves confusion matrix CSV to runs/<experiment>/confusion_matrix_test.csv
  - Optional: saves per-sample predictions CSV (filename, true, pred)
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
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
    start_epoch = ckpt.get("epoch", None)
    return model, start_epoch, ckpt


@torch.no_grad()
def evaluate(model, loader, device="cuda", save_preds_path: Path | None = None):
    model.eval()
    metrics_calc = MetricsCalculator(num_classes=2, class_names=["notdrowsy", "drowsy"])

    rows = []  # for optional predictions export
    all_probs = []  # for ROC curve
    all_labels_for_roc = []
    
    for images, labels, metadata in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        metrics_calc.update(preds, labels)
        
        # Store probabilities and labels for ROC
        all_probs.append(probs.cpu().numpy())
        all_labels_for_roc.append(labels.cpu().numpy())

        if save_preds_path is not None:
            # bring to cpu numpy for serialization
            p_np = preds.cpu().numpy()
            y_np = labels.cpu().numpy()
            prob_np = probs.cpu().numpy()
            for i in range(len(y_np)):
                rows.append({
                    "filename": metadata["filename"][i],
                    "true": int(y_np[i]),
                    "pred": int(p_np[i]),
                    "prob_notdrowsy": float(prob_np[i, 0]),
                    "prob_drowsy": float(prob_np[i, 1])
                })

    metrics = metrics_calc.compute()
    cls_report = metrics_calc.get_classification_report()
    
    # Concatenate all probabilities and labels for ROC
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels_for_roc = np.concatenate(all_labels_for_roc, axis=0)

    if save_preds_path is not None and rows:
        import csv
        with open(save_preds_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "true", "pred", "prob_notdrowsy", "prob_drowsy"])
            writer.writeheader()
            writer.writerows(rows)

    return metrics, cls_report, all_probs, all_labels_for_roc


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pth")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (overrides config)")
    parser.add_argument("--save-preds", action="store_true", help="Save per-sample predictions CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save metrics; default runs/<experiment>")
    parser.add_argument("--save-fig", action="store_true", help="Also save confusion matrix PNGs")
    parser.add_argument("--fig-out-dir", type=str, default="reports/figures", help="Directory to save figure copies")
    args = parser.parse_args()

    # Load config
    config = get_config(args.config)

    # Device
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    print(f"Using device: {device}")

    # Experiment naming/paths
    exp_name = config.get("logging", {}).get("experiment_name", "experiment")
    log_dir = Path(config.get("logging", {}).get("log_dir", "runs"))
    out_dir = Path(args.output_dir) if args.output_dir else (log_dir / exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint path
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        save_dir = Path(config.get("logging", {}).get("save_dir", "checkpoints"))
        ckpt_path = save_dir / f"{exp_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")

    # Data: test only with val/test transforms
    image_size = config.get("data", {}).get("image_size", 224)
    val_transform = get_val_transforms(image_size)

    test_csv = config.get("data", {}).get("test_csv")
    data_root = config.get("data", {}).get("data_root", "datasets/archive")
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

    # Model
    model = create_model(config).to(device)
    model, start_epoch, ckpt = load_checkpoint(model, ckpt_path, device)

    # Evaluate
    preds_csv = out_dir / "predictions_test.csv" if args.save_preds else None
    metrics, cls_report, all_probs, all_labels = evaluate(model, test_loader, device=device, save_preds_path=preds_csv)

    # Print summary
    print("\n==== TEST SET METRICS ====")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:    {metrics['recall_macro']:.4f}")
    print(f"Macro F1:        {metrics['f1_macro']:.4f}")
    
    # Compute and display ROC AUC for each class
    roc_auc_scores = {}
    for class_idx, class_name in enumerate(["notdrowsy", "drowsy"]):
        binary_labels = (all_labels == class_idx).astype(int)
        class_probs = all_probs[:, class_idx]
        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        roc_auc_scores[class_name] = auc(fpr, tpr)
    
    print(f"\nROC AUC Scores:")
    print(f"  notdrowsy: {roc_auc_scores['notdrowsy']:.4f}")
    print(f"  drowsy:    {roc_auc_scores['drowsy']:.4f}")
    print(f"  macro avg: {np.mean(list(roc_auc_scores.values())):.4f}")

    print("\nClassification Report:\n")
    print(cls_report)

    # Save artifacts
    metrics_path = out_dir / "metrics_test.json"
    metrics_to_save = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in metrics.items()}
    metrics_to_save["roc_auc"] = roc_auc_scores
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)

    # Save confusion matrix CSV
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        cm = np.array(cm)
        cm_path = out_dir / "confusion_matrix_test.csv"
        import csv
        with open(cm_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["", "pred_notdrowsy", "pred_drowsy"])
            writer.writerow(["actual_notdrowsy", int(cm[0, 0]), int(cm[0, 1])])
            writer.writerow(["actual_drowsy", int(cm[1, 0]), int(cm[1, 1])])

        # Optionally save PNG heatmaps (raw and normalized)
        if args.save_fig:
            labels = ["notdrowsy", "drowsy"]
            # Raw counts
            fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax1)
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            ax1.set_title('Confusion Matrix (Test)')
            png_path = out_dir / "confusion_matrix_test.png"
            fig1.tight_layout()
            fig1.savefig(png_path, dpi=150)
            plt.close(fig1)

            # Normalized per-actual (row-wise)
            with np.errstate(all='ignore'):
                row_sums = cm.sum(axis=1, keepdims=True)
                cm_norm = cm / np.maximum(row_sums, 1)
            fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax2, vmin=0, vmax=1)
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            ax2.set_title('Confusion Matrix (Normalized)')
            png_norm_path = out_dir / "confusion_matrix_test_normalized.png"
            fig2.tight_layout()
            fig2.savefig(png_norm_path, dpi=150)
            plt.close(fig2)
            
            # ROC curves (one per class + macro average)
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            colors = ['#1f77b4', '#ff7f0e']
            for class_idx, (class_name, color) in enumerate(zip(["notdrowsy", "drowsy"], colors)):
                binary_labels = (all_labels == class_idx).astype(int)
                class_probs = all_probs[:, class_idx]
                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                roc_auc = roc_auc_scores[class_name]
                ax3.plot(fpr, tpr, color=color, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
            
            ax3.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance')
            ax3.set_xlim([0.0, 1.0])
            ax3.set_ylim([0.0, 1.05])
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curves (Test Set)')
            ax3.legend(loc='lower right')
            ax3.grid(alpha=0.3)
            roc_path = out_dir / "roc_curves_test.png"
            fig3.tight_layout()
            fig3.savefig(roc_path, dpi=150)
            plt.close(fig3)

            # Also copy to reports/figures for easy access
            figures_dir = Path(args.fig_out_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
            try:
                import shutil
                shutil.copy2(png_path, figures_dir / png_path.name)
                shutil.copy2(png_norm_path, figures_dir / png_norm_path.name)
                shutil.copy2(roc_path, figures_dir / roc_path.name)
            except Exception as e:
                print(f"Warning: could not copy figures to {figures_dir}: {e}")

    print("\nSaved:")
    print(f"- {metrics_path}")
    if cm is not None:
        print(f"- {cm_path}")
        if args.save_fig:
            print(f"- {out_dir / 'confusion_matrix_test.png'}")
            print(f"- {out_dir / 'confusion_matrix_test_normalized.png'}")
            print(f"- {out_dir / 'roc_curves_test.png'}")
            print(f"- {Path(args.fig_out_dir) / 'confusion_matrix_test.png'}")
            print(f"- {Path(args.fig_out_dir) / 'confusion_matrix_test_normalized.png'}")
            print(f"- {Path(args.fig_out_dir) / 'roc_curves_test.png'}")
    if preds_csv is not None:
        print(f"- {preds_csv}")


if __name__ == "__main__":
    main()
