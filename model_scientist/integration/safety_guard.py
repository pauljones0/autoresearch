"""
SafetyGuard — guardrails against metric proliferation and pipeline overuse.

Enforces limits on:
- Maximum active metrics (default 20)
- Maximum critic proposals per cycle (default 3)
- Mandatory human review for metrics touching the training loop
- Compute budget caps for ablation and multi-scale testing
"""

import time
import json
import os


class SafetyGuard:
    """Pipeline-wide safety guardrails."""

    def __init__(
        self,
        max_active_metrics: int = 20,
        max_critic_proposals_per_cycle: int = 3,
        max_ablation_budget_fraction: float = 0.15,
        max_scale_test_budget_fraction: float = 0.25,
        require_review_for_training_metrics: bool = True,
        state_path: str = None,
    ):
        self.max_active_metrics = max_active_metrics
        self.max_critic_proposals_per_cycle = max_critic_proposals_per_cycle
        self.max_ablation_budget_fraction = max_ablation_budget_fraction
        self.max_scale_test_budget_fraction = max_scale_test_budget_fraction
        self.require_review_for_training_metrics = require_review_for_training_metrics
        self.state_path = state_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "safety_state.json"
        )

        # Runtime state
        self._cycle_proposals = 0
        self._cycle_start = time.time()
        self._total_training_compute = 0.0  # seconds
        self._ablation_compute_used = 0.0
        self._scale_test_compute_used = 0.0
        self._review_queue = []  # metrics pending human review

    def reset_cycle(self):
        """Reset per-cycle counters (call at start of each experiment cycle)."""
        self._cycle_proposals = 0
        self._cycle_start = time.time()

    # --- Metric guardrails ---

    def can_add_metric(self, n_active: int) -> tuple:
        """Check if a new metric can be added.

        Returns:
            (allowed: bool, reason: str)
        """
        if n_active >= self.max_active_metrics:
            return False, f"active metric limit reached ({n_active}/{self.max_active_metrics})"
        return True, "ok"

    def can_propose_metrics(self, n_proposals: int = 1) -> tuple:
        """Check if the critic can propose more metrics this cycle.

        Returns:
            (allowed: bool, max_allowed: int, reason: str)
        """
        remaining = self.max_critic_proposals_per_cycle - self._cycle_proposals
        if remaining <= 0:
            return False, 0, f"proposal limit reached ({self._cycle_proposals}/{self.max_critic_proposals_per_cycle})"
        allowed = min(n_proposals, remaining)
        return True, allowed, f"{remaining} proposals remaining this cycle"

    def record_proposals(self, n: int):
        """Record that n proposals were made this cycle."""
        self._cycle_proposals += n

    def needs_review(self, metric_computation: str) -> bool:
        """Check if a metric's computation method touches the training loop.

        Training-loop metrics contain references to optimizer state, loss
        computation, or gradient manipulation — these need human review.
        """
        if not self.require_review_for_training_metrics:
            return False
        training_keywords = [
            "optimizer", "backward", "grad", "loss.backward",
            "zero_grad", "step()", "param_groups", "learning_rate",
            "lr_schedule", "train_loss", "model.train",
        ]
        code_lower = metric_computation.lower()
        return any(kw in code_lower for kw in training_keywords)

    def queue_for_review(self, metric_name: str, computation: str):
        """Add a metric to the human review queue."""
        self._review_queue.append({
            "metric_name": metric_name,
            "computation": computation,
            "queued_at": time.time(),
            "status": "pending",
        })

    def get_review_queue(self) -> list:
        """Get metrics pending human review."""
        return [r for r in self._review_queue if r["status"] == "pending"]

    def approve_metric(self, metric_name: str):
        """Mark a metric as approved after human review."""
        for r in self._review_queue:
            if r["metric_name"] == metric_name:
                r["status"] = "approved"
                return True
        return False

    def reject_metric(self, metric_name: str):
        """Mark a metric as rejected after human review."""
        for r in self._review_queue:
            if r["metric_name"] == metric_name:
                r["status"] = "rejected"
                return True
        return False

    # --- Compute budget guardrails ---

    def set_training_compute(self, seconds: float):
        """Set the total training compute budget for the session."""
        self._total_training_compute = seconds

    def can_run_ablation(self, estimated_seconds: float) -> tuple:
        """Check if ablation compute budget allows another run.

        Returns:
            (allowed: bool, reason: str)
        """
        if self._total_training_compute <= 0:
            return True, "no training compute baseline set"
        budget = self._total_training_compute * self.max_ablation_budget_fraction
        remaining = budget - self._ablation_compute_used
        if estimated_seconds > remaining:
            return False, (
                f"ablation budget exhausted: {self._ablation_compute_used:.0f}s used "
                f"of {budget:.0f}s ({self.max_ablation_budget_fraction*100:.0f}% of training)"
            )
        return True, f"{remaining:.0f}s ablation budget remaining"

    def record_ablation_compute(self, seconds: float):
        """Record ablation compute usage."""
        self._ablation_compute_used += seconds

    def can_run_scale_test(self, estimated_seconds: float) -> tuple:
        """Check if scale testing compute budget allows another run.

        Returns:
            (allowed: bool, reason: str)
        """
        if self._total_training_compute <= 0:
            return True, "no training compute baseline set"
        budget = self._total_training_compute * self.max_scale_test_budget_fraction
        remaining = budget - self._scale_test_compute_used
        if estimated_seconds > remaining:
            return False, (
                f"scale test budget exhausted: {self._scale_test_compute_used:.0f}s used "
                f"of {budget:.0f}s ({self.max_scale_test_budget_fraction*100:.0f}% of training)"
            )
        return True, f"{remaining:.0f}s scale test budget remaining"

    def record_scale_test_compute(self, seconds: float):
        """Record scale test compute usage."""
        self._scale_test_compute_used += seconds

    # --- Status ---

    def status(self) -> dict:
        """Return full safety guard status."""
        training = self._total_training_compute
        return {
            "cycle_proposals": self._cycle_proposals,
            "max_proposals_per_cycle": self.max_critic_proposals_per_cycle,
            "max_active_metrics": self.max_active_metrics,
            "ablation_compute_used_s": self._ablation_compute_used,
            "ablation_budget_s": training * self.max_ablation_budget_fraction if training > 0 else None,
            "scale_test_compute_used_s": self._scale_test_compute_used,
            "scale_test_budget_s": training * self.max_scale_test_budget_fraction if training > 0 else None,
            "review_queue_size": len(self.get_review_queue()),
        }

    def save_state(self):
        """Persist state to disk."""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        state = {
            "ablation_compute_used": self._ablation_compute_used,
            "scale_test_compute_used": self._scale_test_compute_used,
            "total_training_compute": self._total_training_compute,
            "review_queue": self._review_queue,
            "saved_at": time.time(),
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load persisted state from disk."""
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
            self._ablation_compute_used = state.get("ablation_compute_used", 0.0)
            self._scale_test_compute_used = state.get("scale_test_compute_used", 0.0)
            self._total_training_compute = state.get("total_training_compute", 0.0)
            self._review_queue = state.get("review_queue", [])
        except (json.JSONDecodeError, KeyError):
            pass
