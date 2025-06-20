"""Module for Alzheimer's disease prediction and analysis using neural networks.

this utility module provides functions to load and preprocess data,
define a simple neural network architecture, and train the model acrh and the eval

Author: Orestis Antonis Makris AM 1084516
Date: 2025-4-21
License: MIT
University of Patras, Department of Computer Engineering and Informatics
This code is part of a project for the course "Computational Inteligence".
"""


import numpy as np
import pandas as pd
import torch 

import torch.nn  as nn
import torch.optim as optim
import  matplotlib.pyplot as plt

import seaborn as sns


from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose  import ColumnTransformer

import copy  

def load_data(csv_path):
    """Load and preprocess the data from a CSV file."""
    df = pd.read_csv(csv_path)

    label_col = 'Diagnosis'

    ordinal_cols = ['EducationLevel']
    nominal_cols = []
    binary_cols = [
        'Gender', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease',
        'Diabetes', 'Depression', 'Hypertension', 'HeadInjury',
        'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
        'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
    ]

    all_feature_cols = set(df.columns) - {label_col}
    non_cont_cols = set(ordinal_cols + nominal_cols + binary_cols)
    continuous_cols = sorted(list(all_feature_cols - non_cont_cols))

    missing = all_feature_cols ^ (non_cont_cols | set(continuous_cols))
    if missing:
        raise ValueError(f"Some features not assigned: {missing}")

    X = df[continuous_cols + ordinal_cols + nominal_cols + binary_cols].copy()
    y = df[label_col].values.astype(int)

    return X, y, continuous_cols, ordinal_cols, nominal_cols, binary_cols



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

