import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def plot_confusion_matrix(csv_path, output_path):
    # Read CSV
    # The CSV format from the previous `type` output was:
    # ,notdrowsy,drowsy
    # notdrowsy,7706,1726
    # drowsy,6283,3301
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    plt.figure(figsize=(8, 6))
    sns.set_context('notebook', font_scale=1.2)
    
    # Create heatmap
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.title('Confusion Matrix (MobileNetV3 Regularized - Epoch 6)', fontsize=16, pad=20)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_confusion_matrix.py <csv_path> <output_path>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    output_path = sys.argv[2]
    plot_confusion_matrix(csv_path, output_path)
