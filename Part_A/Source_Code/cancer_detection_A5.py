"""Module for Alzheimer's disease prediction and analysis using neural networks.

This module provides functionality to load, preprocess, train models, and 
visualize results for Alzheimer's disease prediction from patient data.
Author: Orestis Antonis Makris AM 1084516
Date: 2025-3-05
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def load_data(csv_path):
    """Load and preprocess the data from a CSV file.

    Automatically infers continuous columns by subtracting
    ordinal, nominal, and binary columns from the DataFrame's
    set of features.

    Args:
        csv_path: String path to the CSV file.

    Returns:
        tuple: (X, y, continuous_cols, ordinal_cols, nominal_cols, binary_cols)
            X: DataFrame containing feature columns
            y: numpy array of encoded diagnostic labels
            continuous_cols: list of continuous feature names
            ordinal_cols: list of ordinal feature names
            nominal_cols: list of nominal feature names
            binary_cols: list of binary feature names
            
    Raises:
        ValueError: If some features are not properly categorized.
    """
    df = pd.read_csv(csv_path)

    # 1) Drop unwanted columns
    for col in ['DoctorInCharge', 'PatientID']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # 2) Label encode the Diagnosis
    label_col = 'Diagnosis'

    # 3) Declare non-continuous cols
    ordinal_cols = ['EducationLevel']
    nominal_cols = ['Ethnicity']
    binary_cols = [
        'Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'Hypertension', 'HeadInjury',
        'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
        'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
    ]

    # 4) Infer continuous cols
    all_feature_cols = set(df.columns) - {label_col}
    non_cont_cols = set(ordinal_cols + nominal_cols + binary_cols)
    continuous_cols = sorted(list(all_feature_cols - non_cont_cols))

    # Sanity check
    missing = all_feature_cols ^ (non_cont_cols | set(continuous_cols))
    if missing:
        raise ValueError(f"Some features not assigned: {missing}")

    # 5) Split X/y
    X = df[continuous_cols + ordinal_cols + nominal_cols + binary_cols].copy()
    y = df[label_col].values.astype(int)

    return X, y, continuous_cols, ordinal_cols, nominal_cols, binary_cols

class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2, activation=nn.Tanh()):
        super(DeepNet, self).__init__()
        layers = []
        in_feat = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_feat, h))
            layers.append(activation)
            in_feat = h
        layers.append(nn.Linear(in_feat, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, optimizer, criterion, x_train, y_train, x_val, y_val, epochs=50, device=torch.device("cpu"), patience=1, min_delta=1e-2):
    
    """Train the model with early stopping."""

    model.to(device).train()
    train_losses, val_losses = [], []

    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')

    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs+1):
        # training step
        optimizer.zero_grad()
        out = model(x_train.to(device))

        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        preds = out.argmax(1)
        train_accs.append((preds == y_train.to(device)).float().mean().item())

        # validation step
        model.eval()
        with torch.no_grad():
            val_out = model(x_val.to(device))

            val_loss = criterion(val_out, y_val.to(device)).item()

            val_losses.append(val_loss)
            val_preds = val_out.argmax(1)
            val_accs.append((val_preds == y_val.to(device)).float().mean().item())
        model.train()

        # early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, train_accs, val_accs

def evaluate(model, x_test, y_test, device=torch.device("cpu")):
    """Evaluate model performance on test data.
    
    Args:
        model: Trained PyTorch model.
        x_test: Test features tensor.
        y_test: Test labels tenso.
        device: Device to use for evaluation. Defaults to CPU.
        
    Returns:
        float: Accuracy score on test data.
    """
    model.to(device).eval()
    with torch.no_grad():
        out = model(x_test.to(device))

        preds = out.argmax(1).cpu()

    return (preds == y_test).float().mean().item()


def plot_accuracy_curves(train_acc, val_acc, save_path):
    """Plot training and validation accuracy over epochs.
    
    Args:
        train_acc: List of training accuracies.
        val_acc: List of validation accuracies.
        save_path: Path to save the plot.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    epochs = range(1, len(train_acc)+1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, train_acc, label='Train', color='#4ECDC4')
    plt.plot(epochs, val_acc, label='Val', color='#FF6B6B')
    plt.title('Accuracy Over Epochs', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_fold_accuracies(fold_accs, save_path):
    """Plot accuracies across cross-validation folds."""
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    df = pd.DataFrame({
        'Fold': [f"Fold {i+1}" for i in range(len(fold_accs))],
        'Accuracy': fold_accs
    })

    mean_acc = np.mean(fold_accs)
    ax = sns.barplot(x='Fold', y='Accuracy', hue='Fold', data=df, palette="viridis", dodge=False)

    # only remove the legend if it exists
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    for i, acc in enumerate(fold_accs):
        ax.text(i, acc + 0.01, f"{acc:.4f}", ha='center', fontweight='bold')

    plt.axhline(mean_acc, color='r', ls='--', label=f"Mean {mean_acc:.4f}")
    plt.title('CV Fold Accuracies', fontsize=18, fontweight='bold', pad=20)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_mean_loss_curves(mean_train_losses, mean_val_losses, save_path):
    """Plot average training and validation loss across folds.
    
    Args:
        mean_train_losses: List of mean training losses.
        mean_val_losses: List of mean validation losses.
        save_path: Path to save the plot.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    epochs = range(1, len(mean_train_losses) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, mean_train_losses, label='Mean Train Loss', color='#4ECDC4')
    plt.plot(epochs, mean_val_losses, label='Mean Val Loss', color='#FF6B6B')
    plt.title('Mean Loss Over Epochs (All Folds)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mean_accuracy_curves(mean_train_accs, mean_val_accs, save_path):
    """Plot average training and validation accuracy across folds.
    
    Args:
        mean_train_accs: List of mean training accuracies.
        mean_val_accs: List of mean validation accuracies.
        save_path: Path to save the plot.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    epochs = range(1, len(mean_train_accs) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, mean_train_accs, label='Mean Train Acc', color='#4ECDC4')
    plt.plot(epochs, mean_val_accs, label='Mean Val Acc', color='#FF6B6B')
    plt.title('Mean Accuracy Over Epochs (All Folds)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    sns.set_theme(style="whitegrid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_df, y_np, cont_cols, ord_cols, nom_cols, bin_cols = load_data('alzheimers_disease_data.csv')
    pre = ColumnTransformer([
        ('cont', StandardScaler(), cont_cols),
        ('ord',  StandardScaler(), ord_cols),
        ('nom',  OneHotEncoder(sparse=False, handle_unknown='ignore'), nom_cols),
        ('bin',  'passthrough', bin_cols),
    ])
    x_processed = pre.fit_transform(x_df)
    x_tensor = torch.tensor(x_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    architectures = {
        '1-hidden_35':      [35],
        '2-hidden_35-25':   [35, 25],
        '3-hidden_35-70-25':[35, 70, 25],
    }

    print("\nDeepNet Architectures CV Results:")
    print(" Arch                CE_loss±σ    MSE_loss±σ    Acc±σ")

    for name, hsizes in architectures.items():
        ce_list, mse_list, acc_list = [], [], []
        hist_tr_loss, hist_val_loss = [], []
        hist_tr_acc, hist_val_acc = [], []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(x_tensor, y_tensor):
            x_tr, y_tr = x_tensor[train_idx].to(device), y_tensor[train_idx].to(device)
            x_te, y_te = x_tensor[test_idx].to(device),  y_tensor[test_idx].to(device)

            model = DeepNet(x_tr.shape[1], hsizes).to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.6 ,weight_decay=0.001)
            crit = nn.CrossEntropyLoss()

            tr_l, val_l, tr_a, val_a = train(
                model, optimizer, crit,
                x_tr, y_tr, x_te, y_te,
                epochs=200, device=device,
                patience=20, min_delta=1e-3
            )

            hist_tr_loss.append(tr_l);    hist_val_loss.append(val_l)
            hist_tr_acc.append(tr_a);     hist_val_acc.append(val_a)
            ce_list.append(val_l[-1])

            with torch.no_grad():
                probs = torch.softmax(model(x_te), dim=1)
                y_ohe  = nn.functional.one_hot(y_te, num_classes=2).float()
                mse_list.append(nn.MSELoss()(probs, y_ohe).item())

            acc_list.append(evaluate(model, x_te.cpu(), y_te.cpu()))

        mean_ce, std_ce   = np.mean(ce_list),  np.std(ce_list)
        mean_mse, std_mse = np.mean(mse_list), np.std(mse_list)
        mean_acc, std_acc = np.mean(acc_list), np.std(acc_list)

        print(f"{name:<20} {mean_ce:.4f}±{std_ce:.4f}   "
              f"{mean_mse:.4f}±{std_mse:.4f}   "
              f"{mean_acc:.4f}±{std_acc:.4f}")

        # convergence plots
        min_ep = min(len(l) for l in hist_tr_loss)
        mn_tr_loss  = np.mean([l[:min_ep] for l in hist_tr_loss], axis=0)
        mn_val_loss = np.mean([l[:min_ep] for l in hist_val_loss], axis=0)
        plot_mean_loss_curves(mn_tr_loss, mn_val_loss, f"loss_{name}.png")

        min_ep_a    = min(len(a) for a in hist_tr_acc)
        mn_tr_acc   = np.mean([a[:min_ep_a] for a in hist_tr_acc], axis=0)
        mn_val_acc  = np.mean([a[:min_ep_a] for a in hist_val_acc], axis=0)
        plot_mean_accuracy_curves(mn_tr_acc, mn_val_acc, f"acc_{name}.png")

if __name__ == "__main__":
    main()