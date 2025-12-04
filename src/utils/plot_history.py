import json
import matplotlib.pyplot as plt
import glob
import os
import argparse
import pandas as pd
from pathlib import Path

def plot_history(experiment_name, save_dir='checkpoints'):
    save_path = Path(save_dir)
    
    # Find all train and val json files
    train_files = sorted(glob.glob(str(save_path / f"{experiment_name}_train_epoch*.json")))
    val_files = sorted(glob.glob(str(save_path / f"{experiment_name}_val_epoch*.json")))
    
    if not train_files:
        print(f"No log files found for experiment: {experiment_name}")
        return

    train_data = []
    for f in train_files:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            # Extract epoch from filename if not in json
            # Filename format: name_train_epochX.json
            try:
                epoch = int(f.split('epoch')[-1].split('.')[0])
                data['epoch'] = epoch
                train_data.append(data)
            except:
                pass

    val_data = []
    for f in val_files:
        with open(f, 'r') as json_file:
            data = json.load(json_file)
            try:
                epoch = int(f.split('epoch')[-1].split('.')[0])
                data['epoch'] = epoch
                val_data.append(data)
            except:
                pass
    
    # Convert to dataframe
    df_train = pd.DataFrame(train_data).sort_values('epoch')
    df_val = pd.DataFrame(val_data).sort_values('epoch')
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(df_train['epoch'], df_train['loss'], label='Train', marker='o')
    if not df_val.empty:
        axes[0].plot(df_val['epoch'], df_val['loss'], label='Val', marker='o')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    if 'accuracy' in df_train.columns:
        axes[1].plot(df_train['epoch'], df_train['accuracy'], label='Train', marker='o')
        if not df_val.empty and 'accuracy' in df_val.columns:
            axes[1].plot(df_val['epoch'], df_val['accuracy'], label='Val', marker='o')
        axes[1].set_title('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
    # F1 Score
    if 'f1_macro' in df_train.columns:
        axes[2].plot(df_train['epoch'], df_train['f1_macro'], label='Train', marker='o')
        if not df_val.empty and 'f1_macro' in df_val.columns:
            axes[2].plot(df_val['epoch'], df_val['f1_macro'], label='Val', marker='o')
        axes[2].set_title('Macro F1-Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    output_file = save_path / f"{experiment_name}_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='fast_mobilenet', help='Experiment name')
    parser.add_argument('--dir', type=str, default='checkpoints', help='Checkpoints directory')
    args = parser.parse_args()
    
    plot_history(args.experiment, args.dir)
