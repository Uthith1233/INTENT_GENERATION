from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, random_split


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_intent_pipeline"
DATA_PATH = OUTPUT_DIR / "lstm_trajectory_windows.npz"
INDEX_PATH = OUTPUT_DIR / "lstm_traj_window_index.csv"
MODEL_PATH = OUTPUT_DIR / "social_lstm_intent_model.pt"

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 5.0
USE_WEIGHTED_SAMPLER = True
CLASS_WEIGHT_POWER = 0.0
BEST_SCORE_NAME = "macro_f1"
EVENT_CONFIDENCE_THRESHOLDS = [
    0.0,
    0.3,
    0.5,
    0.6,
    0.7,
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TrajectoryWindowDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.X_neighbors = torch.tensor(data["X_neighbors"], dtype=torch.float32)
        self.neighbor_mask = torch.tensor(data["neighbor_mask"], dtype=torch.bool)
        self.y = torch.tensor(data["y"], dtype=torch.long)

        self.feature_cols = data["feature_cols"]
        self.neighbor_feature_cols = data["neighbor_feature_cols"]
        self.label_names = data["label_names"]

        self.feature_cols_list = [str(col) for col in self.feature_cols.tolist()]
        self.neighbor_feature_cols_list = [
            str(col) for col in self.neighbor_feature_cols.tolist()
        ]
        self.label_names_list = [str(label) for label in self.label_names.tolist()]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_neighbors[idx], self.neighbor_mask[idx], self.y[idx]


class SocialLSTMIntentModel(nn.Module):
    def __init__(
        self,
        target_input_size,
        neighbor_input_size,
        hidden_size,
        num_layers,
        num_classes,
    ):
        super().__init__()

        self.target_norm = nn.LayerNorm(target_input_size)
        self.neighbor_norm = nn.LayerNorm(neighbor_input_size)

        self.target_lstm = nn.LSTM(
            input_size=target_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.social_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.social_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x, x_neighbors, neighbor_mask):
        # x shape: [batch, history_frames, feature_dim]
        # x_neighbors shape: [batch, history_frames, max_neighbors, neighbor_dim]
        x = self.target_norm(x)
        x_neighbors = self.neighbor_norm(x_neighbors)

        target_out, _ = self.target_lstm(x)

        neighbor_emb = self.neighbor_encoder(x_neighbors)
        neighbor_mask_float = neighbor_mask.unsqueeze(-1).float()
        neighbor_emb = neighbor_emb * neighbor_mask_float

        target_for_neighbors = target_out.unsqueeze(2).expand(
            -1,
            -1,
            neighbor_emb.size(2),
            -1,
        )
        attention_input = torch.cat([target_for_neighbors, neighbor_emb], dim=-1)
        attention_scores = self.social_attention(attention_input).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~neighbor_mask, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=2).unsqueeze(-1)
        attention_weights = attention_weights * neighbor_mask_float
        social_context = (neighbor_emb * attention_weights).sum(dim=2)

        has_neighbor = neighbor_mask.any(dim=2, keepdim=True)
        social_context = torch.where(
            has_neighbor,
            social_context,
            torch.zeros_like(social_context),
        )

        gate_input = torch.cat([target_out, social_context], dim=-1)
        gated_social_context = self.social_gate(gate_input) * social_context

        fused = torch.cat([target_out, gated_social_context], dim=-1)
        fused = self.fusion(fused)
        fused = fused + target_out

        last_hidden = fused[:, -1, :]

        logits = self.classifier(last_hidden)

        return logits


def compute_class_weights(y, num_classes, power=0.5):
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp(min=1.0)
    weights = weights.pow(power)
    weights = weights / weights.mean()
    return weights


def make_weighted_sampler(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    sample_weights = 1.0 / counts.clamp(min=1.0)[y]
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def predict_from_logits(logits, event_threshold=0.0):
    probs = torch.softmax(logits, dim=1)
    confidence, preds = probs.max(dim=1)

    if event_threshold > 0.0:
        preds = preds.clone()
        low_confidence_event = (preds != 0) & (confidence < event_threshold)
        preds[low_confidence_event] = 0

    return preds


def evaluate_logits(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total = 0

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, X_neighbors, neighbor_mask, y in loader:
            X = X.to(DEVICE)
            X_neighbors = X_neighbors.to(DEVICE)
            neighbor_mask = neighbor_mask.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X, X_neighbors, neighbor_mask)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            total += X.size(0)

            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / max(total, 1)

    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    return avg_loss, all_logits, all_targets


def evaluate(model, loader, criterion, event_threshold=0.0):
    avg_loss, logits, targets = evaluate_logits(model, loader, criterion)
    preds = predict_from_logits(logits, event_threshold=event_threshold)
    acc = (preds == targets).float().mean().item()

    return avg_loss, acc, preds, targets


def compute_metrics(preds, targets, num_classes):
    confusion = torch.bincount(
        targets * num_classes + preds,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()

    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
    f1 = torch.where(
        precision + recall > 0,
        2.0 * precision * recall / (precision + recall),
        torch.zeros_like(tp),
    )

    present_classes = support > 0
    present_event_classes = present_classes.clone()
    if num_classes > 0:
        present_event_classes[0] = False

    total = support.sum().item()
    accuracy = tp.sum().item() / max(total, 1.0)
    macro_f1 = f1[present_classes].mean().item() if present_classes.any() else 0.0
    event_macro_f1 = (
        f1[present_event_classes].mean().item()
        if present_event_classes.any()
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "event_macro_f1": event_macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.long(),
        "confusion": confusion,
    }


def choose_best_threshold(logits, targets, thresholds, score_name, num_classes):
    best_threshold = 0.0
    best_score = -1.0
    best_metrics = None
    best_preds = None

    for threshold in thresholds:
        preds = predict_from_logits(logits, event_threshold=threshold)
        metrics = compute_metrics(preds, targets, num_classes)
        score = metrics[score_name]

        if (
            score > best_score
            or (
                score == best_score
                and best_metrics is not None
                and metrics["accuracy"] > best_metrics["accuracy"]
            )
        ):
            best_threshold = threshold
            best_score = score
            best_metrics = metrics
            best_preds = preds

    return best_threshold, best_score, best_metrics, best_preds


def print_label_distribution(name, y, label_names):
    counts = torch.bincount(y, minlength=len(label_names))
    print(f"\n{name} label distribution:")

    for class_id, label in enumerate(label_names):
        print(f"{label:25s}: {int(counts[class_id])}")


def print_detailed_metrics(preds, targets, label_names):
    metrics = compute_metrics(preds, targets, len(label_names))

    print("\nValidation metrics:")
    print(f"accuracy      : {metrics['accuracy']:.4f}")
    print(f"macro_f1      : {metrics['macro_f1']:.4f}")
    print(f"event_macro_f1: {metrics['event_macro_f1']:.4f}")

    print("\nPer-class metrics:")

    for class_id, label in enumerate(label_names):
        count = int(metrics["support"][class_id])

        if count == 0:
            print(f"{label:25s}: no samples")
            continue

        precision = metrics["precision"][class_id].item()
        recall = metrics["recall"][class_id].item()
        f1 = metrics["f1"][class_id].item()

        print(
            f"{label:25s}: "
            f"precision={precision:.4f} | "
            f"recall={recall:.4f} | "
            f"f1={f1:.4f} | "
            f"count={count}"
        )

    short_names = [str(label)[:8] for label in label_names]
    print("\nConfusion matrix rows=true cols=pred:")
    print("true\\pred".ljust(14) + "".join(f"{name:>10s}" for name in short_names))

    for class_id, label in enumerate(short_names):
        row = metrics["confusion"][class_id]
        print(f"{label:14s}" + "".join(f"{int(value):10d}" for value in row))


def get_subset_labels(dataset, subset):
    if isinstance(subset, Subset):
        return dataset.y[subset.indices]

    return dataset.y


def make_train_val_split(dataset, index_path, train_fraction=0.8, seed=42):
    if not index_path.exists():
        num_samples = len(dataset)
        train_size = int(train_fraction * num_samples)
        val_size = num_samples - train_size

        return random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed),
        )

    index_df = pd.read_csv(index_path)

    if len(index_df) != len(dataset):
        raise ValueError(
            f"Index length {len(index_df)} does not match dataset length {len(dataset)}"
        )

    rng = np.random.default_rng(seed)
    scenes = np.array(sorted(index_df["scene_name"].dropna().unique()))
    rng.shuffle(scenes)

    split_idx = max(1, int(train_fraction * len(scenes)))
    split_idx = min(split_idx, len(scenes) - 1)

    train_scenes = set(scenes[:split_idx])

    train_indices = index_df.index[index_df["scene_name"].isin(train_scenes)].tolist()
    val_indices = index_df.index[~index_df["scene_name"].isin(train_scenes)].tolist()

    print(f"Scene split: {len(train_scenes)} train scenes, {len(scenes) - len(train_scenes)} val scenes")
    print(f"Window split: {len(train_indices)} train windows, {len(val_indices)} val windows")

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing LSTM dataset: {DATA_PATH}. Run test_file_mini.py first."
        )

    dataset = TrajectoryWindowDataset(DATA_PATH)

    print("Loaded dataset")
    print("X shape:", dataset.X.shape)
    print("X_neighbors shape:", dataset.X_neighbors.shape)
    print("neighbor_mask shape:", dataset.neighbor_mask.shape)
    print("y shape:", dataset.y.shape)
    print("Features:", dataset.feature_cols_list)
    print("Neighbor features:", dataset.neighbor_feature_cols_list)
    print("Labels:", dataset.label_names_list)

    train_ds, val_ds = make_train_val_split(dataset, INDEX_PATH)

    input_size = dataset.X.shape[-1]
    neighbor_input_size = dataset.X_neighbors.shape[-1]
    num_classes = len(dataset.label_names_list)

    train_y = get_subset_labels(dataset, train_ds)
    val_y = get_subset_labels(dataset, val_ds)

    print_label_distribution("Train", train_y, dataset.label_names_list)
    print_label_distribution("Validation", val_y, dataset.label_names_list)

    train_sampler = (
        make_weighted_sampler(train_y, num_classes)
        if USE_WEIGHTED_SAMPLER
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = SocialLSTMIntentModel(
        target_input_size=input_size,
        neighbor_input_size=neighbor_input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
    ).to(DEVICE)

    class_weights = compute_class_weights(
        train_y,
        num_classes,
        power=CLASS_WEIGHT_POWER,
    ).to(DEVICE)

    print("Class weights:", class_weights.detach().cpu().numpy())
    print("Weighted sampler:", USE_WEIGHTED_SAMPLER)
    print("Optimizer: AdamW")
    print("Weight decay:", WEIGHT_DECAY)
    print("Gradient clip norm:", GRAD_CLIP_NORM)
    print("Best checkpoint score:", BEST_SCORE_NAME)
    print("Event confidence thresholds:", EVENT_CONFIDENCE_THRESHOLDS)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_score = -1.0
    best_val_acc = 0.0
    best_event_threshold = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for X, X_neighbors, neighbor_mask, y in train_loader:
            X = X.to(DEVICE)
            X_neighbors = X_neighbors.to(DEVICE)
            neighbor_mask = neighbor_mask.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X, X_neighbors, neighbor_mask)
            loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            preds = logits.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            correct += (preds == y).sum().item()
            total += X.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_logits, val_targets = evaluate_logits(
            model,
            val_loader,
            criterion,
        )
        val_threshold, val_score, val_metrics, val_preds = choose_best_threshold(
            logits=val_logits,
            targets=val_targets,
            thresholds=EVENT_CONFIDENCE_THRESHOLDS,
            score_name=BEST_SCORE_NAME,
            num_classes=num_classes,
        )
        val_acc = val_metrics["accuracy"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"event_macro_f1={val_metrics['event_macro_f1']:.4f} | "
            f"event_threshold={val_threshold:.2f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_acc = val_acc
            best_event_threshold = val_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_cols": dataset.feature_cols_list,
                    "neighbor_feature_cols": dataset.neighbor_feature_cols_list,
                    "label_names": dataset.label_names_list,
                    "input_size": input_size,
                    "neighbor_input_size": neighbor_input_size,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "num_classes": num_classes,
                    "model_type": "SocialAttentionGatedLSTMIntentModel",
                    "weight_decay": WEIGHT_DECAY,
                    "grad_clip_norm": GRAD_CLIP_NORM,
                    "best_score_name": BEST_SCORE_NAME,
                    "best_score": best_val_score,
                    "best_val_acc": best_val_acc,
                    "event_confidence_threshold": best_event_threshold,
                },
                MODEL_PATH,
            )

    print(f"\nBest validation {BEST_SCORE_NAME}:", best_val_score)
    print("Best validation accuracy at that checkpoint:", best_val_acc)
    print("Best event confidence threshold:", best_event_threshold)

    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_event_threshold = float(
            checkpoint.get("event_confidence_threshold", best_event_threshold)
        )

    val_loss, val_logits, val_targets = evaluate_logits(
        model,
        val_loader,
        criterion,
    )
    val_preds = predict_from_logits(
        val_logits,
        event_threshold=best_event_threshold,
    )

    print("\nFinal evaluation event confidence threshold:", best_event_threshold)

    print_detailed_metrics(
        preds=val_preds,
        targets=val_targets,
        label_names=dataset.label_names_list,
    )


if __name__ == "__main__":
    main()
