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
