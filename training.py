from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


DATA_PATH = Path("outputs_intent_pipeline/lstm_trajectory_windows.npz")

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrajectoryWindowDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

        self.feature_cols = data["feature_cols"]
        self.label_names = data["label_names"]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleLSTMIntentModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x shape: [batch, history_frames, feature_dim]
        out, _ = self.lstm(x)

        # Use last timestep hidden state.
        last_hidden = out[:, -1, :]

        logits = self.classifier(last_hidden)

        return logits


def compute_class_weights(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts + 1.0)
    weights = weights / weights.mean()
    return weights


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += X.size(0)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return avg_loss, acc, all_preds, all_targets


def print_per_class_accuracy(preds, targets, label_names):
    print("\nPer-class accuracy:")

    for class_id, label in enumerate(label_names):
        mask = targets == class_id

        if mask.sum() == 0:
            print(f"{label:25s}: no samples")
            continue

        acc = (preds[mask] == targets[mask]).float().mean().item()

        print(f"{label:25s}: {acc:.4f} | count={int(mask.sum())}")


def main():
    dataset = TrajectoryWindowDataset(DATA_PATH)

    print("Loaded dataset")
    print("X shape:", dataset.X.shape)
    print("y shape:", dataset.y.shape)
    print("Features:", list(dataset.feature_cols))
    print("Labels:", list(dataset.label_names))

    num_samples = len(dataset)
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    input_size = dataset.X.shape[-1]
    num_classes = len(dataset.label_names)

    model = SimpleLSTMIntentModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    ).to(DEVICE)

    class_weights = compute_class_weights(dataset.y, num_classes).to(DEVICE)

    print("Class weights:", class_weights.detach().cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += X.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, val_preds, val_targets = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cols": dataset.feature_cols,
                    "label_names": dataset.label_names,
                    "input_size": input_size,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": num_classes,
                },
                "outputs_intent_pipeline/simple_lstm_intent_model.pt",
            )

    print("\nBest validation accuracy:", best_val_acc)

    val_loss, val_acc, val_preds, val_targets = evaluate(
        model,
        val_loader,
        criterion,
    )

    print_per_class_accuracy(
        preds=val_preds,
        targets=val_targets,
        label_names=dataset.label_names,
    )


if __name__ == "__main__":
    main()