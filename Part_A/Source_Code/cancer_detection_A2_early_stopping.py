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


class SimpleNet(nn.Module):
    """A simple neural network with one hidden layer and custom activation.
    
    Attributes:
        fc1: First fully connected layer.
        activation: Activation function to use.
        fc2: Second fully connected layer.
    """
    
    def __init__(self, input_size, hidden_size=35, output_size=2, activation=nn.Tanh()):
        """Initialize the neural network.
        
        Args:
            input_size: Integer size of the input features.
            hidden_size: Integer size of the hidden layer. Defaults to 32.
            output_size: Integer size of the output layer. Defaults to 2.
            activation: PyTorch activation function. Defaults to ReLU.
        """
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after passing through the network.
        """
        out = self.activation(self.fc1(x))
        return self.fc2(out)


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
    """Run the main workflow for model training and evaluation."""
    sns.set_theme(style="whitegrid")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split into feature-groups
    x_df, y_np, cont_cols, ord_cols, nom_cols, bin_cols = load_data('alzheimers_disease_data.csv')

   

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('cont', StandardScaler(), cont_cols),
        ('ord', StandardScaler(), ord_cols),
        ('nom', OneHotEncoder(sparse=False, handle_unknown='ignore'), nom_cols),
        ('bin', 'passthrough', bin_cols),
    ])

    # Apply transforms
    x_processed = preprocessor.fit_transform(x_df)

    # Plot AFTER scaling for continuous+ordinal using same function
    num_cols = cont_cols + ord_cols
    df_scaled = pd.DataFrame(x_processed[:, :len(num_cols)], columns=num_cols)

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    # Cross-validation setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


    fold_accs = []

    best_acc = 0.0
    best_model = None
    
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []

    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(x_tensor, y_tensor), 1):
        x_train, x_test = x_tensor[train_idx], x_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

        model = SimpleNet(x_train.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_losses, val_losses, train_accs, val_accs = train(
            model, optimizer, criterion,
            x_train, y_train, x_test, y_test,
            epochs=200, device=device,
            patience=3,     # early stopping patience
            min_delta=1e-3   # ελάχιστη βελτίωση
        )
        
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_train_accs.append(train_accs)
        all_val_accs.append(val_accs)

        plot_accuracy_curves(train_accs, val_accs, f"accuracy_fold{fold_idx}.png")

        acc = evaluate(model, x_test, y_test, device=device)
        fold_accs.append(acc)
        print(f"Fold {fold_idx} Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model

    # Print cross-validation results
    print(f"\nCV Mean Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    plot_fold_accuracies(fold_accs, "fold_accuracies.png")

    min_ep = min(len(lst) for lst in all_train_losses)
    mean_train_losses = np.mean([lst[:min_ep] for lst in all_train_losses], axis=0)
    mean_val_losses   = np.mean([lst[:min_ep] for lst in all_val_losses],   axis=0)
    plot_mean_loss_curves(mean_train_losses, mean_val_losses, "mean_loss_curve.png")

    min_ep_acc      = min(len(lst) for lst in all_train_accs)
    mean_train_accs = np.mean([lst[:min_ep_acc] for lst in all_train_accs], axis=0)
    mean_val_accs   = np.mean([lst[:min_ep_acc] for lst in all_val_accs],   axis=0)
    plot_mean_accuracy_curves(mean_train_accs, mean_val_accs, "mean_accuracy_curve.png")


if __name__ == "__main__":
    main()