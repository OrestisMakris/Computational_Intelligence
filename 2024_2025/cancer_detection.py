import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(csv_path):
    """Load and preprocess the data from a CSV file.

    Reads the CSV file, drops unused columns, encodes labels,
    and separates numerical and categorical features.

    Args:
        csv_path (str): Path to the CSV data file.

    Returns:
        X (pd.DataFrame): Features data.
        y (np.ndarray): Encoded label array.
        numeric_cols (list): List of numeric feature column names.
        categorical_cols (list): List of categorical feature column names.
    """
    df = pd.read_csv(csv_path)

    # Drop columns that will not be used (like DoctorInCharge)
    if 'DoctorInCharge' in df.columns:
        df = df.drop('DoctorInCharge', axis=1)

    # Define categorical columns (remove duplicate entries)
    categorical_cols = ['Gender', 'Ethnicity', 'Smoking',
                        'FamilyHistoryAlzheimers', 'Diabetes',
                        'Depression', 'HeadInjury', 'Hypertension']
    label_col = 'Diagnosis'

    # Remove PatientID if not used.
    if 'PatientID' in df.columns:
        df = df.drop('PatientID', axis=1)

    # Identify numeric columns by excluding categorical and label columns.
    numeric_cols = [col for col in df.columns if col not in categorical_cols + [label_col]]

    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    df[label_col] = label_encoder.fit_transform(df[label_col])

    # Extract features and labels
    X = df.drop(label_col, axis=1)
    y = df[label_col].values.astype(int)

    return X, y, numeric_cols, categorical_cols


def plot_feature_distribution(df, numeric_cols, save_prefix):
    """Plot distribution histograms for numeric features.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        numeric_cols (list): List of numeric columns to plot.
        save_prefix (str): Prefix for the saved plot filenames.
    """
    for col in numeric_cols:
        plt.figure()
        plt.hist(df[col].dropna(), bins=30)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col}')
        plt.savefig(f"{save_prefix}_{col}.png")
        plt.close()


def plot_categorical_distribution(df, categorical_cols, save_prefix):
    """Plot bar charts for categorical feature distributions.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        categorical_cols (list): List of categorical columns to plot.
        save_prefix (str): Prefix for the saved plot filenames.
    """
    for col in categorical_cols:
        if col in df.columns:
            plt.figure()
            counts = df[col].value_counts()
            plt.bar(counts.index.astype(str), counts.values)
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.title(f'Categorical Distribution of {col}')
            plt.savefig(f"{save_prefix}_{col}.png")
            plt.close()


class SimpleNet(nn.Module):
    """A simple neural network with one hidden layer supporting a custom activation."""
    
    def __init__(self, input_size, hidden_size=32, output_size=2, activation=nn.ReLU()):
        """Initialize the SimpleNet model.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in hidden layer.
            output_size (int): Number of output classes.
            activation (nn.Module): Activation function to use.
        """
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass of the network.
        
        Args:
            x (Tensor): Input data tensor.
            
        Returns:
            Tensor: Output tensor.
        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out


def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=50, device=torch.device("cpu")):
    """Train the network and record loss history.

    Args:
        model (nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (nn.Module): Loss function.
        X_train (Tensor): Training inputs.
        y_train (Tensor): Training labels.
        X_val (Tensor): Validation inputs.
        y_val (Tensor): Validation labels.
        epochs (int): Number of training epochs.
        device (torch.device): Device for computation (CPU/CUDA).

    Returns:
        list, list: Training and validation loss histories.
    """
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Evaluate on validation set.
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            val_losses.append(val_loss.item())
        model.train()
    return train_losses, val_losses


def evaluate(model, X_test, y_test, device=torch.device("cpu")):
    """Evaluate the model and return accuracy.

    Args:
        model (nn.Module): The neural network model.
        X_test (Tensor): Test inputs.
        y_test (Tensor): Test labels.
        device (torch.device): Device for computation.

    Returns:
        float: Accuracy score.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.cpu() == y_test).float().mean().item()
    return accuracy


def main():
    """Main function to perform data loading, preprocessing, training, and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    csv_path = 'alzheimers_disease_data.csv'  # update path if necessary
    X_df, y_np, numeric_cols, categorical_cols = load_data(csv_path)

    # Plot distributions before applying any transformations.
    plot_feature_distribution(X_df, numeric_cols, save_prefix="raw_numeric")
    plot_categorical_distribution(X_df, categorical_cols, save_prefix="raw_categorical")

    # Define the transformation pipeline.
    numeric_transformer = StandardScaler()  # Alternatives: MinMaxScaler(), RobustScaler()
    categorical_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply transformation.
    X_processed = preprocessor.fit_transform(X_df)

    # Plot distributions after transformation (numeric part).
    X_numeric_scaled = X_processed[:, :len(numeric_cols)]
    for idx, col in enumerate(numeric_cols):
        plt.figure()
        plt.hist(X_numeric_scaled[:, idx], bins=30)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {col} after Standard Scaling')
        plt.savefig(f"scaled_{col}.png")
        plt.close()

    # Convert processed data to a torch tensor.
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.long)

    # Plot class distribution.
    y_unique, counts = np.unique(y_np, return_counts=True)
    plt.figure()
    plt.bar(y_unique, counts, tick_label=[f'Class {c}' for c in y_unique])
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.savefig("class_distribution_processed.png")
    plt.close()

    # Define activation functions to test.
    activations = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "SiLU": nn.SiLU()  # SiLU (Swish) is available in recent versions of PyTorch.
    }

    # Use StratifiedKFold cross-validation.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for act_name, act_func in activations.items():
        print(f"\nTesting activation function: {act_name}")
        fold_accuracies = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, y_tensor), start=1):
            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

            model = SimpleNet(input_size=X_train.shape[1],hidden_size=32, output_size=2, activation=act_func).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_losses, val_losses = train(
                model, optimizer, criterion, X_train, y_train, X_test, y_test,
                epochs=250, device=device)

            accuracy = evaluate(model, X_test, y_test, device=device)
            fold_accuracies.append(accuracy)
            print(f'Fold {fold} Accuracy: {accuracy:.4f}')

            # Export loss curves for the first fold.
            if fold == 1:
                plt.figure()
                plt.plot(train_losses, label='Training Loss')
                plt.plot(val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Training vs Validation Loss (Fold 1) - {act_name}')
                plt.legend()
                plt.savefig(f"loss_curves_fold1_{act_name}.png")
                plt.close()

        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        print(f'Mean Accuracy for {act_name}: {mean_accuracy:.4f}')


if __name__ == '__main__':
    main()