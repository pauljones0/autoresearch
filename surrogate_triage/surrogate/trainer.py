"""
Phase 2 — SurrogateTrainer: lightweight MLP surrogate model in PyTorch
for predicting val_bpb delta from enriched feature vectors.
"""

import copy
import json
import logging
import math
import os
import random

import torch
import torch.nn as nn

from surrogate_triage.schemas import SurrogateTrainingExample, SurrogatePrediction

logger = logging.getLogger(__name__)


class SurrogateMLP(nn.Module):
    """Small MLP: input_dim -> 32 -> 16 -> 1 (kept under 10K params)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SurrogateTrainer:
    """Train and use a surrogate MLP model.

    The model predicts the expected val_bpb delta for a candidate
    modification, given its enriched feature vector.
    """

    def __init__(self, input_dim: int = 279, device: str | None = None):
        """
        Args:
            input_dim: Dimension of the input feature vector
                       (default 256 code + 23 failure = 279, plus any metrics).
            device: Torch device string. Auto-detects if None.
        """
        self.input_dim = input_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: SurrogateMLP | None = None
        self._train_mean: list[float] | None = None
        self._train_std: list[float] | None = None

    def train(
        self,
        examples: list[SurrogateTrainingExample],
        epochs: int = 100,
        lr: float = 1e-3,
        k_folds: int = 5,
        patience: int = 10,
        seed: int = 42,
    ) -> dict:
        """Train the surrogate model with k-fold cross-validation.

        Args:
            examples: Training examples with feature_vector and actual_delta.
            epochs: Maximum training epochs per fold.
            lr: Adam learning rate.
            k_folds: Number of cross-validation folds.
            patience: Early stopping patience (epochs).
            seed: Random seed for reproducibility.

        Returns:
            Dict with training metrics (avg_mse, avg_mae, fold_results).
        """
        if len(examples) < 5:
            raise ValueError(f"Need at least 5 examples, got {len(examples)}")

        random.seed(seed)
        torch.manual_seed(seed)

        # Prepare data
        features = [ex.feature_vector for ex in examples]
        labels = [ex.actual_delta for ex in examples]

        # Detect input_dim from data
        self.input_dim = len(features[0])

        # Compute normalization stats
        self._compute_normalization(features)

        # K-fold cross-validation
        indices = list(range(len(examples)))
        random.shuffle(indices)
        fold_size = len(indices) // k_folds
        fold_results = []

        best_model_state = None
        best_val_mse = float("inf")

        for fold in range(k_folds):
            start = fold * fold_size
            end = start + fold_size if fold < k_folds - 1 else len(indices)
            val_idx = set(indices[start:end])
            train_idx = [i for i in indices if i not in val_idx]

            train_X = self._to_tensor([features[i] for i in train_idx])
            train_y = torch.tensor([labels[i] for i in train_idx], dtype=torch.float32).to(self.device)
            val_X = self._to_tensor([features[i] for i in val_idx])
            val_y = torch.tensor([labels[i] for i in val_idx], dtype=torch.float32).to(self.device)

            model = SurrogateMLP(self.input_dim).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            best_fold_loss = float("inf")
            epochs_no_improve = 0
            best_fold_state = None

            for epoch in range(epochs):
                # Training
                model.train()
                optimizer.zero_grad()
                pred = model(train_X)
                loss = criterion(pred, train_y)
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_pred = model(val_X)
                    val_loss = criterion(val_pred, val_y).item()

                if val_loss < best_fold_loss - 1e-6:
                    best_fold_loss = val_loss
                    best_fold_state = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    break

            # Evaluate best fold model
            if best_fold_state is not None:
                model.load_state_dict(best_fold_state)
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X).cpu().tolist()
                val_actual = val_y.cpu().tolist()

            fold_mse = sum((p - a) ** 2 for p, a in zip(val_pred, val_actual)) / len(val_actual)
            fold_mae = sum(abs(p - a) for p, a in zip(val_pred, val_actual)) / len(val_actual)

            fold_results.append({
                "fold": fold,
                "val_mse": fold_mse,
                "val_mae": fold_mae,
                "n_train": len(train_idx),
                "n_val": len(list(val_idx)),
            })

            if fold_mse < best_val_mse:
                best_val_mse = fold_mse
                best_model_state = copy.deepcopy(model.state_dict())

        # Final model: retrain on all data with best architecture
        self.model = SurrogateMLP(self.input_dim).to(self.device)
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Also do a final full training pass
        self._train_full(features, labels, epochs=epochs, lr=lr, patience=patience)

        avg_mse = sum(f["val_mse"] for f in fold_results) / len(fold_results)
        avg_mae = sum(f["val_mae"] for f in fold_results) / len(fold_results)

        metrics = {
            "avg_mse": avg_mse,
            "avg_mae": avg_mae,
            "k_folds": k_folds,
            "n_examples": len(examples),
            "input_dim": self.input_dim,
            "fold_results": fold_results,
        }

        logger.info(
            "Training complete: avg_mse=%.6f, avg_mae=%.6f, n=%d",
            avg_mse, avg_mae, len(examples),
        )
        return metrics

    def predict(self, feature_vector: list[float]) -> SurrogatePrediction:
        """Predict delta for a single feature vector.

        Args:
            feature_vector: Enriched feature vector.

        Returns:
            SurrogatePrediction with predicted_delta and confidence.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        self.model.eval()
        x = self._to_tensor([feature_vector])
        with torch.no_grad():
            pred = self.model(x).item()

        # Simple confidence: inverse of distance from training mean prediction
        confidence = min(1.0, max(0.0, 1.0 - abs(pred) * 0.5))

        return SurrogatePrediction(
            predicted_delta=pred,
            confidence=confidence,
            adjusted_score=pred,
        )

    def feature_importance(
        self,
        examples: list[SurrogateTrainingExample],
        n_repeats: int = 5,
    ) -> list[tuple[int, float]]:
        """Permutation-based feature importance.

        Shuffles each feature dimension, measures accuracy drop (MSE increase).

        Args:
            examples: Test examples.
            n_repeats: Number of permutation repeats per feature.

        Returns:
            List of (feature_index, importance_score) sorted descending.
        """
        if self.model is None:
            raise RuntimeError("Model not trained.")

        features = [ex.feature_vector for ex in examples]
        labels = [ex.actual_delta for ex in examples]

        X = self._to_tensor(features)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            baseline_pred = self.model(X)
            baseline_mse = nn.MSELoss()(baseline_pred, y).item()

        importances = []
        for dim in range(self.input_dim):
            total_increase = 0.0
            for _ in range(n_repeats):
                X_perm = X.clone()
                perm = torch.randperm(X_perm.size(0))
                X_perm[:, dim] = X_perm[perm, dim]
                with torch.no_grad():
                    perm_pred = self.model(X_perm)
                    perm_mse = nn.MSELoss()(perm_pred, y).item()
                total_increase += perm_mse - baseline_mse
            importances.append((dim, total_increase / n_repeats))

        importances.sort(key=lambda x: x[1], reverse=True)
        return importances

    def save(self, path: str):
        """Save model weights and metadata to disk."""
        if self.model is None:
            raise RuntimeError("No model to save.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "train_mean": self._train_mean,
            "train_std": self._train_std,
        }, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str):
        """Load model weights and metadata from disk."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.input_dim = checkpoint["input_dim"]
        self._train_mean = checkpoint.get("train_mean")
        self._train_std = checkpoint.get("train_std")

        self.model = SurrogateMLP(self.input_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded from %s (input_dim=%d)", path, self.input_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_normalization(self, features: list[list[float]]):
        """Compute per-feature mean and std for normalization."""
        n = len(features)
        dim = len(features[0])
        means = [0.0] * dim
        for f in features:
            for i, v in enumerate(f):
                means[i] += v
        means = [m / n for m in means]

        stds = [0.0] * dim
        for f in features:
            for i, v in enumerate(f):
                stds[i] += (v - means[i]) ** 2
        stds = [math.sqrt(s / max(n - 1, 1)) for s in stds]

        # Avoid division by zero
        stds = [s if s > 1e-8 else 1.0 for s in stds]

        self._train_mean = means
        self._train_std = stds

    def _to_tensor(self, features: list[list[float]]) -> torch.Tensor:
        """Convert features to a normalized tensor."""
        t = torch.tensor(features, dtype=torch.float32).to(self.device)
        if self._train_mean is not None and self._train_std is not None:
            mean_t = torch.tensor(self._train_mean, dtype=torch.float32).to(self.device)
            std_t = torch.tensor(self._train_std, dtype=torch.float32).to(self.device)
            t = (t - mean_t) / std_t
        return t

    def _train_full(
        self,
        features: list[list[float]],
        labels: list[float],
        epochs: int,
        lr: float,
        patience: int,
    ):
        """Train model on full dataset with early stopping on training loss."""
        if self.model is None:
            self.model = SurrogateMLP(self.input_dim).to(self.device)

        X = self._to_tensor(features)
        y = torch.tensor(labels, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - 1e-7:
                best_loss = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break
