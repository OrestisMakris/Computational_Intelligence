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
    plt.figure()
    plt.hist(X[:, feature_index].numpy(), bins=30)
    plt.xlabel(f'Feature {feature_index}')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Feature {feature_index}')
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(y, save_path="class_distribution.png"):
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

# Modified SimpleNet accepts a custom activation and customizable hidden size.
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2, activation=nn.ReLU()):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

def train(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=50, device=torch.device("cpu"), patience=10):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    wait = 0  # counter for consecutive epochs without improvement
    
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
        
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    return train_losses, val_losses

def evaluate(model, X_test, y_test, device=torch.device("cpu")):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted.cpu() == y_test).float().mean().item()
    return accuracy

# Compute MSE between softmax-probabilities and one-hot encoded true labels.
def compute_mse(outputs, y, device=torch.device("cpu")):
    probs = torch.softmax(outputs, dim=1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, y.unsqueeze(1).to(device), 1.0)
    mse_loss = ((probs - onehot) ** 2).mean().item()
    return mse_loss

def regularization_experiments(X_scaled, y, device, best_hidden_size):
    """
    Run experiments using L2 regularization (weight decay) on the best topology.
    Evaluate three different values for r: 0.0001, 0.001, 0.01, using 5fold CV.
    For each experiment, convergence plots (CE Loss vs Epochs) for the first fold are saved.
    """
    input_dim = X_scaled.shape[1]
    r_values = [0.0001, 0.001, 0.01]
    results = {}
    epochs = 250
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for r in r_values:
        key = f"r_{r}"
        print(f"\n--- Testing regularization with r = {r} ---")
        ce_losses_total = []
        mse_total = []
        acc_total = []
        fold_num = 0
        
        for train_index, test_index in skf.split(X_scaled, y):
            fold_num += 1
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model = SimpleNet(input_size=input_dim, hidden_size=best_hidden_size, activation=nn.ReLU()).to(device)
            criterion = nn.CrossEntropyLoss()
            # Use Adam with weight_decay (L2 regularization)
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=r)
            
            train_losses, val_losses = train(model, optimizer, criterion, X_train, y_train, X_test, y_test,
                                             epochs=epochs, device=device, patience=10)
            
            model.eval()
            with torch.no_grad():
                outputs = model(X_test.to(device))
                final_ce_loss = criterion(outputs, y_test.to(device)).item()
                mse_loss = compute_mse(outputs, y_test, device)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted.cpu() == y_test).float().mean().item()
            
            ce_losses_total.append(final_ce_loss)
            mse_total.append(mse_loss)
            acc_total.append(accuracy)
            
            if fold_num == 1:
                plt.figure()
                plt.plot(train_losses, label='Training CE Loss')
                plt.plot(val_losses, label='Validation CE Loss')
                plt.xlabel('Epoch')
                plt.ylabel('CE Loss')
                plt.title(f'Convergence Plot (r = {r})')
                plt.legend()
                plt.savefig(f"loss_curves_regularization_r_{r}.png")
                plt.close()
                
        avg_ce = sum(ce_losses_total) / len(ce_losses_total)
        avg_mse = sum(mse_total) / len(mse_total)
        avg_acc = sum(acc_total) / len(acc_total)
        results[key] = {"CE Loss": avg_ce, "MSE": avg_mse, "Accuracy": avg_acc}
        print(f"Results for r = {r}: Avg CE Loss: {avg_ce:.4f}, Avg MSE: {avg_mse:.4f}, Avg Accuracy: {avg_acc:.4f}")
        
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    csv_path = 'alzheimers_disease_data.csv'  # update path if needed
    X, y = load_data(csv_path)
    
    # Export dataset plots (raw data)
    plot_dataset(X, feature_index=0, save_path="dataset_feature_distribution.png")
    plot_class_distribution(y, save_path="class_distribution.png")
    plot_feature_correlation(X, save_path="feature_correlation_heatmap.png")
    
    # Standardize using z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Assume the best topology from Α3: we choose best_hidden_size (for example, I = number of features)
    best_hidden_size = X_scaled.shape[1]
    
    # Run the regularization experiments.
    results_reg = regularization_experiments(X_scaled, y, device, best_hidden_size)
    
    print("\n--- Regularization Experiment Results ---")
    for key, metrics in results_reg.items():
        print(f"{key}: {metrics}")

if __name__ == '__main__':
    main()