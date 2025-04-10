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
        header = next(reader)  # skip header
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

# Updated DeepNet: includes BatchNorm and Dropout to help generalization.
class DeepNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=2, activation=nn.ReLU(), dropout_prob=0.2):
        """
        hidden_sizes: list of integers, one per hidden layer.
        dropout_prob: dropout probability to reduce overfitting.
        """
        super(DeepNet, self).__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h
        # Output layer (no activation because CrossEntropyLoss expects logits)
        layers.append(nn.Linear(in_dim, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Updated train function: now includes a learning rate scheduler and increased epochs.
def train(model, optimizer, scheduler, criterion, X_train, y_train, X_val, y_val, epochs=300, device=torch.device("cpu"), patience=10):
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    wait = 0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            val_losses.append(val_loss.item())
        model.train()
        
        # Step the scheduler using the current validation loss.
        scheduler.step(val_loss.item())
        
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

def compute_mse(outputs, y, device=torch.device("cpu")):
    probs = torch.softmax(outputs, dim=1)
    onehot = torch.zeros_like(probs)
    onehot.scatter_(1, y.unsqueeze(1).to(device), 1.0)
    mse_loss = ((probs - onehot) ** 2).mean().item()
    return mse_loss

def deep_net_experiments(X_scaled, y, device):
    input_dim = X_scaled.shape[1]
    # Original architectures plus huge ones:
    architectures = {
        "2_layers_constant": [input_dim, input_dim],
        "2_layers_decreasing": [int(1.5 * input_dim), input_dim],
        "3_layers_uniform": [input_dim, input_dim, input_dim],
        "3_layers_decreasing": [int(1.5 * input_dim), input_dim, int(0.75 * input_dim)],
        # HUGE architectures:
        "3_layers_huge": [input_dim * 4, input_dim * 4, input_dim * 4],
        "4_layers_huge": [input_dim * 4, input_dim * 4, input_dim * 4, input_dim * 4],
        "5_layers_huge": [input_dim * 4, input_dim * 4, input_dim * 4, input_dim * 4, input_dim * 4]
    }
    results = {}
    epochs = 300
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for arch_label, hidden_sizes in architectures.items():
        print(f"\n--- Testing architecture: {arch_label} with hidden sizes {hidden_sizes} ---")
        ce_losses_total = []
        mse_total = []
        acc_total = []
        fold_num = 0
        
        for train_index, test_index in skf.split(X_scaled, y):
            fold_num += 1
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Create model with dropout and batch normalization.
            model = DeepNet(input_size=input_dim, hidden_sizes=hidden_sizes, activation=nn.ReLU(), dropout_prob=0.2).to(device)
            # Use Adam optimizer with weight decay (L2 regularization).
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            # Scheduler to reduce LR on plateau.
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            criterion = nn.CrossEntropyLoss()
            
            train_losses, val_losses = train(model, optimizer, scheduler, criterion, X_train, y_train, X_test, y_test,
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
                plt.title(f'Convergence Plot ({arch_label})')
                plt.legend()
                plt.savefig(f"loss_curves_deep_{arch_label}.png")
                plt.close()
                
        avg_ce = sum(ce_losses_total) / len(ce_losses_total)
        avg_mse = sum(mse_total) / len(mse_total)
        avg_acc = sum(acc_total) / len(acc_total)
        results[arch_label] = {"CE Loss": avg_ce, "MSE": avg_mse, "Accuracy": avg_acc}
        print(f"Results for {arch_label}: Avg CE Loss: {avg_ce:.4f}, Avg MSE: {avg_mse:.4f}, Avg Accuracy: {avg_acc:.4f}")
    
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    csv_path = "alzheimers_disease_data.csv"  # update path if needed
    X, y = load_data(csv_path)
    
    plot_dataset(X, feature_index=0, save_path="dataset_feature_distribution.png")
    plot_class_distribution(y, save_path="class_distribution.png")
    plot_feature_correlation(X, save_path="feature_correlation_heatmap.png")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.numpy())
    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    
    results_deep = deep_net_experiments(X_scaled, y, device)
    print("\n--- Deep Network Experiment Results ---")
    for arch, metrics in results_deep.items():
        print(f"{arch}: {metrics}")

if __name__ == "__main__":
    main()