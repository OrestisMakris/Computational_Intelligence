"""Module for Alzheimer's disease prediction and analysis using neural networks.

This module contains helper functions for loading data, performing Genetech Algorithm (GA) 
operations for feature selection,preprocessing it, and evaluating the performance of neural network models.
Author: Orestis Antonis Makris AM 1084516
Date: 2025-4-21
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

def load_data(csv_path):
    """Load all features from the CSV file without categorization.

    Args:
        csv_path: String path to the CSV file.

    Returns:
        tuple: (X, y)
            X: DataFrame containing all feature columns.
            y: numpy array of encoded diagnostic labels.
    """
    df = pd.read_csv(csv_path)

    # Drop unwanted columns
    for col in ['DoctorInCharge', 'PatientID']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Label encode the Diagnosis
    label_col = 'Diagnosis'

    # Split X and y
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].values.astype(int)

    return X, y


class SimpleNet(nn.Module):
    """A neural network with three hidden layers and custom activation.
    
    Attributes:
        fc1: First fully connected layer (input to first hidden layer).
        fc2: Second fully connected layer (first hidden to second hidden layer).
        fc3: Third fully connected layer (second hidden to third hidden layer).
        activation: Activation function to use.
        output: Output fully connected layer (third hidden to output layer).
    """
    
    def __init__(self, input_size, activation=nn.Tanh()):
        """Initialize the neural network with three hidden layers.
        
        Args:
            input_size: Integer size of the input features.
            activation: PyTorch activation function. Defaults to Tanh.
        """
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 35)  # First hidden layer with 35 neurons
        self.fc2 = nn.Linear(35, 70)          # Second hidden layer with 70 neurons
        self.fc3 = nn.Linear(70, 25)          # Third hidden layer with 25 neurons
        self.activation = activation
        self.output = nn.Linear(25, 2)        # Output layer with 2 neurons

    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after passing through the network.
        """
        x = self.activation(self.fc1(x))  # Pass through first hidden layer
        x = self.activation(self.fc2(x))  # Pass through second hidden layer
        x = self.activation(self.fc3(x))  # Pass through third hidden layer
        return self.output(x)             # Pass through output layer


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

        preds = out.argmax(1).cpu()
        train_accs.append((preds == y_train).float().mean().item())

        model.eval()
        with torch.no_grad():
            val_out = model(x_val.to(device))
            val_loss = criterion(val_out, y_val.to(device))
            val_losses.append(val_loss.item())
            val_preds = val_out.argmax(1).cpu()
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
        preds = out.argmax(1).cpu()
    return (preds == y_test).float().mean().item()


def plot_numeric_distributions_subplot(df, numeric_cols, save_path):
    """Plot histograms for numeric features.
    
    Args:
        df: DataFrame containing the data.
        numeric_cols: List of numeric column names to plot.
        save_path: Path to save the plot.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    n = len(numeric_cols)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows*4))
    axes = axes.flatten()
    palette = sns.color_palette("viridis", n)
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col].dropna(), bins=30, kde=True, color=palette[i], ax=axes[i])
        axes[i].set_title(f'Distribution of {col}', fontweight='bold')
        mean, std = df[col].mean(), df[col].std()
        axes[i].axvline(mean, color='red', ls='--', label=f"Mean {mean:.2f}")
        axes[i].legend()
        
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle('Feature Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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
    """Plot accuracies across cross-validation folds.
    
    Args:
        fold_accs: List of accuracies for each fold.
        save_path: Path to save the plot.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    df = pd.DataFrame({
        'Fold': [f"Fold {i+1}" for i in range(len(fold_accs))],
        'Accuracy': fold_accs
    })
    
    mean_acc = np.mean(fold_accs)
    ax = sns.barplot(x='Fold', y='Accuracy', data=df, palette="viridis")
    
    for i, acc in enumerate(fold_accs):
        ax.text(i, acc+0.01, f"{acc:.4f}", ha='center', fontweight='bold')
        
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
    plot_numeric_distributions_subplot(df_scaled, num_cols, "feature_distributions_after.png")

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
            model, optimizer, criterion, x_train, y_train, x_test, y_test,
            epochs=200, device=device
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

    # Mean loss/accuracy curves across folds
    mean_train_losses = np.mean(all_train_losses, axis=0)
    mean_val_losses = np.mean(all_val_losses, axis=0)
    plot_mean_loss_curves(mean_train_losses, mean_val_losses, "mean_loss_curve.png")

    mean_train_accs = np.mean(all_train_accs, axis=0)
    mean_val_accs = np.mean(all_val_accs, axis=0)
    plot_mean_accuracy_curves(mean_train_accs, mean_val_accs, "mean_accuracy_curve.png")


if __name__ == "__main__":
    main()


def initialize_population(pop_size, num_features):      
    """ Δημιουργεί αρχικό πληθυσμό: τυχαία δυαδικά διανύσματα, 
        με τουλάχιστον ένα bit = 1."""
    
    population = []
    for _ in range(pop_size):
        individual = np.random.randint(0, 2, size=num_features)
        # Ensure at least one feature is selected
        if np.sum(individual) == 0:
            individual[np.random.randint(0, num_features)] = 1
        population.append(individual)
    return np.array(population)
       

