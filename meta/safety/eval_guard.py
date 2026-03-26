"""
Evaluation metric guard — verify evaluation code unchanged via SHA-256.
"""

import hashlib
import os
from meta.schemas import BoundaryViolationError


class EvaluationMetricGuard:
    """SHA-256 hash verification of evaluation code."""

    def __init__(self, train_path: str = "train.py",
                 state_path: str = "meta_state.json"):
        self.train_path = train_path
        self.state_path = state_path
        self._stored_hash: str = ""

    def _compute_eval_hash(self) -> str:
        try:
            with open(self.train_path, "r", encoding="utf-8") as f:
                source = f.read()
        except FileNotFoundError:
            return "file_not_found"
        return hashlib.sha256(source.encode()).hexdigest()

    def initialize(self):
        self._stored_hash = self._compute_eval_hash()

    def verify_evaluation_unchanged(self) -> bool:
        if not self._stored_hash:
            self.initialize()
            return True
        current_hash = self._compute_eval_hash()
        if current_hash != self._stored_hash:
            raise BoundaryViolationError(
                "evaluation_modified",
                f"Eval hash changed: {self._stored_hash[:16]}... -> {current_hash[:16]}...",
            )
        return True
