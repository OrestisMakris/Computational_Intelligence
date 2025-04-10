import csv
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def load_data(csv_path):
    data = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # skip header line if present
        for row in reader:
            # Remove the last column ("DoctorInCharge")
            row = row[:-1]
            data.append([float(x) for x in row])
    data = torch.tensor(data)
    X = data[:, :-1]
    y = data[:, -1].long()
    return X, y

def plot_dataset(X, feature_index=0, save_path="dataset_feature_distribution.png"):
    # Plot distribution for a chosen feature and export the figure
    plt.figure()
    plt.hist(X[:, feature_index].numpy(), bins=30)
    plt.xlabel(f'Feature {feature_index}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Feature {feature_index}')
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(y, save_path="class_distribution.png"):
    # Plot a bar chart for the distribution of classes in y
    y_np = y.numpy()
    classes, counts = np.unique(y_np, return_counts=True)
    plt.figure()
    plt.bar(classes, counts, tick_label=[f'Class {c}' for c in classes])
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.savefig(save_path)
    plt.close()

def plot_feature_correlation(X, save_path="feature_correlation_heatmap.png"):
    # Calculate correlation matrix and plot a heatmap
    X_np = X.numpy()
    corr_matrix = np.corrcoef(X_np, rowvar=False)
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.savefig(save_path)
    plt.close()

# Modified SimpleNet class accepts a custom activation function.
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size=32, output_size=2, activation=nn.ReLU()):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=50, device=torch.device("cpu")):
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
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            val_losses.append(val_loss.item())
        model.train()
    return train_losses, val_losses

def evaluate(model, X_test, y_test, device=torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.cpu() == y_test).float().mean().item()
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    csv_path = 'alzheimers_disease_data.csv'  # update path if necessary
    X, y = load_data(csv_path)

    # Export dataset plots (using raw data)
    plot_dataset(X, feature_index=0, save_path="dataset_feature_distribution.png")
    plot_class_distribution(y, save_path="class_distribution.png")
    plot_feature_correlation(X, save_path="feature_correlation_heatmap.png")
    
    # Standardization using z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)

    # Define the activation functions to test:
    activations = {
        "ReLU": nn.ReLU(),
        "Tanh": nn.Tanh(),
        "SiLU": nn.SiLU()  # SiLU (Swish) is available in recent versions of PyTorch
    }
    
    # Loop over each activation to train and evaluate the network
    for act_name, act_func in activations.items():
        print(f"\nTesting activation function: {act_name}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
    
        # For demonstration, record loss curves for the first fold only.
        for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Pass the tested activation into the network
            model = SimpleNet(input_size=X_train.shape[1], activation=act_func).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train the model and record loss curves; adjust epochs if needed.
            train_losses, val_losses = train(model, optimizer, criterion, X_train, y_train, X_test, y_test, epochs=250, device=device)
            acc = evaluate(model, X_test, y_test, device=device)
            accuracies.append(acc)
            print(f'Fold {fold} Accuracy: {acc:.4f}')
            
            # Export loss curves for the first fold for each activation
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
            
        mean_acc = sum(accuracies)/len(accuracies)
        print(f'Mean Accuracy for {act_name}: {mean_acc:.4f}')

if __name__ == '__main__':
    main()