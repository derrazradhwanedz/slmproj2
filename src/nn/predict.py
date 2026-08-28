"""Inference-time question -> answer quality-profile predictor.

Fixes the training/inference mismatch found in the old main.py: predict()
applies the same fitted MinMaxScaler used during training (instead of
clipping raw metrics into [0, 1]) and inverse-transforms the network's
output back to each metric's natural scale before it is used to build an
MGCoT prompt.
"""

from typing import Dict, List

import joblib
import numpy as np
import torch

from metrics import profile as profile_metrics
from nn.model import QAPredictorConfig, QAPredictorNN

METRIC_COLS: List[str] = profile_metrics.__all__
SCALED_RANGE = (0.0, 1.0)  # MinMaxScaler's fitted output range


class QualityProfilePredictor:
    """Loads a trained QAPredictorNN and its fitted scalers for inference."""

    def __init__(
        self,
        model_path: str,
        q_scaler_path: str,
        a_scaler_path: str,
        device: str = "cpu",
    ) -> None:
        """Load the trained model and scalers.

        Args:
            model_path: Path to the saved model state_dict (.pth).
            q_scaler_path: Path to the fitted question MinMaxScaler (.pkl).
            a_scaler_path: Path to the fitted answer MinMaxScaler (.pkl).
            device: Torch device to run inference on ("cpu" or "cuda").

        Raises:
            FileNotFoundError: If any of the three artifacts is missing.
        """
        self.device = torch.device(device)
        self.model = QAPredictorNN(QAPredictorConfig(num_metrics=len(METRIC_COLS))).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.q_scaler = joblib.load(q_scaler_path)
        self.a_scaler = joblib.load(a_scaler_path)
        self.metrics = [getattr(profile_metrics, name)() for name in METRIC_COLS]

    def predict(self, question: str) -> Dict[str, float]:
        """Predict the target answer quality profile for a question.

        Args:
            question: The raw question text.

        Returns:
            A dict mapping each metric name to its predicted value, on that
            metric's natural scale (not the internal [0, 1] training scale).

        Raises:
            TypeError: If question is not a string.
        """
        if not isinstance(question, str):
            raise TypeError(f"question must be a str, got {type(question).__name__}")

        raw_q_metrics = np.array([[metric(question) for metric in self.metrics]])
        q_scaled = self.q_scaler.transform(raw_q_metrics)

        with torch.no_grad():
            input_tensor = torch.FloatTensor(q_scaled).to(self.device)
            a_scaled = self.model(input_tensor).cpu().numpy()

        a_scaled = np.clip(a_scaled, *SCALED_RANGE)
        a_raw = self.a_scaler.inverse_transform(a_scaled)[0]
        return dict(zip(METRIC_COLS, a_raw))
