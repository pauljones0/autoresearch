"""
Structured logging for meta-experiments.

Appends typed entries to meta_log.jsonl for audit trail and recovery.
Entry types:
  - meta_experiment_started
  - meta_experiment_completed
  - meta_posterior_update
  - meta_promotion
  - meta_regime_change
  - meta_budget_status
"""

import json
import time


class MetaExperimentLogger:
    """Append structured entries to meta_log.jsonl."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def log(self, entry_type: str, data: dict = None) -> None:
        """Append a single typed entry to the log file.

        Args:
            entry_type: One of the defined entry types.
            data: Additional fields to include in the entry.
        """
        entry = {"type": entry_type, "timestamp": time.time()}
        if data:
            entry.update(data)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_experiment_started(self, experiment_id: str,
                               config_diff: list,
                               meta_iteration: int) -> None:
        self.log("meta_experiment_started", {
            "experiment_id": experiment_id,
            "config_diff": config_diff,
            "meta_iteration": meta_iteration,
        })

    def log_experiment_completed(self, experiment_id: str,
                                 improvement_rate: float,
                                 compared_to_baseline: str,
                                 meta_iteration: int,
                                 n_iterations: int) -> None:
        self.log("meta_experiment_completed", {
            "experiment_id": experiment_id,
            "improvement_rate": improvement_rate,
            "compared_to_baseline": compared_to_baseline,
            "meta_iteration": meta_iteration,
            "n_iterations": n_iterations,
        })

    def log_posterior_update(self, param_id: str, variant_key: str,
                            alpha: float, beta: float,
                            zone: str) -> None:
        self.log("meta_posterior_update", {
            "param_id": param_id,
            "variant_key": variant_key,
            "alpha": alpha,
            "beta": beta,
            "zone": zone,
        })

    def log_promotion(self, param_id: str, promoted_value,
                      config_update: dict) -> None:
        self.log("meta_promotion", {
            "param_id": param_id,
            "promoted_value": promoted_value,
            "config_update": config_update,
        })

    def log_regime_change(self, old_regime: str, new_regime: str,
                          reason: str = "") -> None:
        self.log("meta_regime_change", {
            "old_regime": old_regime,
            "new_regime": new_regime,
            "reason": reason,
        })

    def log_budget_status(self, budget_used: float,
                          budget_fraction: float,
                          total_meta_experiments: int) -> None:
        self.log("meta_budget_status", {
            "budget_used": budget_used,
            "budget_fraction": budget_fraction,
            "total_meta_experiments": total_meta_experiments,
        })
