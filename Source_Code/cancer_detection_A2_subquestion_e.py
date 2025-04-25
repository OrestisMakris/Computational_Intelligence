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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch.nn.functional as F

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
    
    def __init__(self, input_size, hidden_size=32, output_size=2, activation=nn.Tanh()):
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


def train(model, optimizer, criterion, x_train, y_train, x_val, y_val, 
          epochs=50, device=torch.device("cpu")):
    """Train the neural network model.
    
    Args:
        model: PyTorch model to train.
        optimizer: PyTorch optimizer to use.
        criterion: Loss function.
        x_train: Training features tensor.
        y_train: Training labels tensor.
        x_val: Validation features tensor.
        y_val: Validation labels tensor.
        epochs: Number of training epochs. Defaults to 50.
        device: Device to use for training. Defaults to CPU.
        
    Returns:
        tuple: (train_losses, val_losses, train_accs, val_accs)
            Lists containing loss and accuracy values for each epoch.
    """
    model.to(device).train()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(x_train.to(device))
        loss = criterion(out, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        preds = out.argmax(1)
        train_accs.append((preds == y_train).float().mean().item())

        model.eval()
        with torch.no_grad():
            val_out = model(x_val.to(device))
            val_loss = criterion(val_out, y_val.to(device))
            val_losses.append(val_loss.item())
            val_preds = val_out.argmax(1)
            val_accs.append((val_preds == y_val).float().mean().item())
        model.train()
        
    return train_losses, val_losses, train_accs, val_accs


def evaluate(model, x_test, y_test, device=torch.device("cpu")):
    """Evaluate model performance on test data.
    
    Args:
        model: Trained PyTorch model.
        x_test: Test features tensor.
        y_test: Test labels tensor.
        device: Device to use for evaluation. Defaults to CPU.
        
    Returns:
        float: Accuracy score on test data.
    """
    model.to(device).eval()
    with torch.no_grad():
        out = model(x_test.to(device))
        preds = out.argmax(1)
    return (preds == y_test).float().mean().item()




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


    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    input_size = x_tensor.shape[1]
    # Candidate hidden sizes in [I/2, 2I]
    hidden_sizes = [input_size // 2,2 * input_size // 3,input_size,2 * input_size]
    # Activation for hidden layer
    hidden_activation = nn.Tanh()

    table_results = []
    for H in hidden_sizes:
        fold_ce, fold_mse, fold_acc = [], [], []
        all_tr_losses, all_val_losses = [], []
        all_tr_accs, all_val_accs = [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(x_tensor, y_tensor), 1):
            x_tr = x_tensor[tr_idx].to(device);  y_tr = y_tensor[tr_idx].to(device)
            x_te = x_tensor[te_idx].to(device);  y_te = y_tensor[te_idx].to(device)

            model = SimpleNet(input_size, hidden_size=H,
                  output_size=2, activation=hidden_activation).to(device)
            crit_ce  = nn.CrossEntropyLoss()
            crit_mse = nn.MSELoss()
            opt= optim.Adam(model.parameters(), lr=1e-3)

            tr_l, val_l, tr_a, val_a = train(
                model, opt, crit_ce, x_tr, y_tr, x_te, y_te,
               epochs=200, device=device
            )
            all_tr_losses.append(tr_l);  all_val_losses.append(val_l)
            all_tr_accs.append(tr_a);     all_val_accs.append(val_a)

            # final CE val loss and MSE on softmax probabilities
            with torch.no_grad():
                logits = model(x_te)
                probs  = F.softmax(logits, dim=1)
                y_ohe  = F.one_hot(y_te, num_classes=2).float()
                mse_l  = crit_mse(probs, y_ohe).item()
            fold_ce.append(val_l[-1])
            fold_mse.append(mse_l)
            fold_acc.append(evaluate(model, x_te, y_te, device=device))

         # Plot mean convergence curves for this H
        mn_tr = np.mean(all_tr_losses, axis=0)
        mn_val = np.mean(all_val_losses, axis=0)
        plot_mean_loss_curves(mn_tr, mn_val, f"mean_loss_H{H}.png")
        ma_tr = np.mean(all_tr_accs, axis=0)
        ma_val = np.mean(all_val_accs, axis=0)
        plot_mean_accuracy_curves(ma_tr, ma_val, f"mean_acc_H{H}.png")

        table_results.append({
            'H': H,
            'CE':  f"{np.mean(fold_ce):.4f}±{np.std(fold_ce):.4f}",
            'MSE': f"{np.mean(fold_mse):.4f}±{np.std(fold_mse):.4f}",
            'Acc': f"{np.mean(fold_acc):.4f}±{np.std(fold_acc):.4f}"
        })

    # Display summary table
    print("H    |   CE Loss       |   MSE Loss      |   Accuracy")
    for r in table_results:
        print(f"{r['H']:>4} | {r['CE']:>13} | {r['MSE']:>13} | {r['Acc']:>9}")


if __name__ == "__main__":
    main()