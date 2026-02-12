import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from preprocess import load_data, add_rul, scale_features, make_sequences
from model import LSTMRegressor


class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_FD001.txt")

    # Load and label data
    df = load_data(TRAIN_PATH)
    df = add_rul(df)

    # Feature columns
    feature_cols = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

    # Scale features
    df, scaler = scale_features(df, feature_cols)

    # Create sequences
    X, y = make_sequences(df, feature_cols, seq_len=30, rul_cap=130)

    dataset = RULDataset(X, y)

    # Train/validation split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    model = LSTMRegressor(input_size=len(feature_cols)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def evaluate(loader):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                total_loss += loss.item() * xb.size(0)
        return total_loss / len(loader.dataset)

    # Training loop
    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = evaluate(val_loader)

        print(f"Epoch {epoch:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # Save trained model
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "lstm_rul_fd001.pt")
    torch.save(model.state_dict(), model_path)

    print("Saved model to:", model_path)
    print("Training complete.")


if __name__ == "__main__":
    main()



