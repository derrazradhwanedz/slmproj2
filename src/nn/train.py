"""Training script for QAPredictorNN.

Reads precomputed question/answer quality-profile CSVs (one column per
metric in metrics.profile), fits per-column MinMax scalers, trains
the network, and persists the model weights together with the fitted
scalers so inference can apply the exact same transform.

Run directly as: python src/nn/train.py --q-df ... --a-df ... --model-out ... --q-scaler-out ... --a-scaler-out ...
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from metrics import profile as profile_metrics
from nn.model import QAPredictorConfig, QAPredictorNN
from utils import load_yaml

METRIC_COLS: List[str] = profile_metrics.__all__
_TRAIN_DEFAULTS = load_yaml("nn.yaml")["training"]


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters, kept separate from model architecture.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        test_size: Fraction of data held out for validation.
        seed: Random seed for the train/validation split.
        grad_clip_norm: Max gradient norm for clip_grad_norm_.
        val_check_interval: Run/report validation every N epochs.
    """

    epochs: int = _TRAIN_DEFAULTS["epochs"]
    batch_size: int = _TRAIN_DEFAULTS["batch_size"]
    lr: float = _TRAIN_DEFAULTS["lr"]
    weight_decay: float = _TRAIN_DEFAULTS["weight_decay"]
    test_size: float = _TRAIN_DEFAULTS["test_size"]
    seed: int = _TRAIN_DEFAULTS["seed"]
    grad_clip_norm: float = _TRAIN_DEFAULTS["grad_clip_norm"]
    val_check_interval: int = _TRAIN_DEFAULTS["val_check_interval"]


class QADataset(Dataset):
    """Paired (question profile, answer profile) tensors."""

    def __init__(self, q: np.ndarray, a: np.ndarray) -> None:
        """Store the scaled question/answer metric arrays as tensors.

        Args:
            q: Scaled question metrics, shape (n, num_metrics).
            a: Scaled answer metrics, shape (n, num_metrics).
        """
        self.q = torch.FloatTensor(q)
        self.a = torch.FloatTensor(a)

    def __len__(self) -> int:
        return len(self.q)

    def __getitem__(self, idx: int):
        return self.q[idx], self.a[idx]


def train(
    q_df_path: str,
    a_df_path: str,
    model_out_path: str,
    q_scaler_out_path: str,
    a_scaler_out_path: str,
    model_config: Optional[QAPredictorConfig] = None,
    train_config: Optional[TrainConfig] = None,
) -> QAPredictorNN:
    """Train QAPredictorNN and persist the model plus its fitted scalers.

    Args:
        q_df_path: CSV with one column per METRIC_COLS entry, profiling questions.
        a_df_path: CSV with one column per METRIC_COLS entry, profiling gold answers.
        model_out_path: Where to save the trained model state_dict (.pth).
        q_scaler_out_path: Where to save the fitted question MinMaxScaler (.pkl).
        a_scaler_out_path: Where to save the fitted answer MinMaxScaler (.pkl).
        model_config: Model architecture hyperparameters. Defaults to QAPredictorConfig().
        train_config: Training hyperparameters. Defaults to TrainConfig().

    Returns:
        The trained model, in eval mode.

    Raises:
        ValueError: If either CSV is missing a required metric column.
    """
    model_config = model_config or QAPredictorConfig()
    train_config = train_config or TrainConfig()

    q_df = pd.read_csv(q_df_path)
    a_df = pd.read_csv(a_df_path)

    missing_q = set(METRIC_COLS) - set(q_df.columns)
    missing_a = set(METRIC_COLS) - set(a_df.columns)
    if missing_q or missing_a:
        raise ValueError(
            f"Missing metric columns - question: {missing_q}, answer: {missing_a}"
        )

    min_len = min(len(q_df), len(a_df))
    q_metrics = q_df[METRIC_COLS].fillna(0).values[:min_len]
    a_metrics = a_df[METRIC_COLS].fillna(0).values[:min_len]

    q_scaler = MinMaxScaler()
    a_scaler = MinMaxScaler()
    q_scaled = q_scaler.fit_transform(q_metrics)
    a_scaled = a_scaler.fit_transform(a_metrics)

    x_train, x_val, y_train, y_val = train_test_split(
        q_scaled, a_scaled, test_size=train_config.test_size, random_state=train_config.seed
    )
    train_loader = DataLoader(
        QADataset(x_train, y_train), train_config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        QADataset(x_val, y_val), train_config.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QAPredictorNN(model_config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )

    for epoch in range(train_config.epochs):
        model.train()
        for q, a in train_loader:
            optimizer.zero_grad()
            pred = model(q.to(device))
            loss = criterion(pred, a.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()

        if epoch % train_config.val_check_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss = sum(
                    criterion(model(q.to(device)), a.to(device)).item()
                    for q, a in val_loader
                ) / len(val_loader)
            print(f"Epoch {epoch}: val_mse={val_loss:.4f}")

    model.eval()
    for out_path in (model_out_path, q_scaler_out_path, a_scaler_out_path):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    joblib.dump(q_scaler, q_scaler_out_path)
    joblib.dump(a_scaler, a_scaler_out_path)
    return model


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone training runs."""
    parser = argparse.ArgumentParser(description="Train QAPredictorNN.")
    parser.add_argument("--q-df", required=True, help="Path to the question quality-profile CSV.")
    parser.add_argument("--a-df", required=True, help="Path to the answer quality-profile CSV.")
    parser.add_argument("--model-out", required=True, help="Output path for the model .pth file.")
    parser.add_argument("--q-scaler-out", required=True, help="Output path for the question scaler .pkl file.")
    parser.add_argument("--a-scaler-out", required=True, help="Output path for the answer scaler .pkl file.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        q_df_path=args.q_df,
        a_df_path=args.a_df,
        model_out_path=args.model_out,
        q_scaler_out_path=args.q_scaler_out,
        a_scaler_out_path=args.a_scaler_out,
        train_config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr),
    )
