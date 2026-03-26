"""
Phase 2 — SurrogateRetrainer: trigger surrogate retraining when enough
new journal data has accumulated, with rollback on degradation.
"""

import logging
import os
import shutil

from surrogate_triage.surrogate.journal_data_extractor import JournalDataExtractor
from surrogate_triage.surrogate.feature_enricher import FeatureEnricher
from surrogate_triage.surrogate.trainer import SurrogateTrainer
from surrogate_triage.surrogate.evaluator import SurrogateEvaluator

logger = logging.getLogger(__name__)


class SurrogateRetrainer:
    """Manage periodic retraining of the surrogate model.

    Triggers retraining after every M new datapoints. Compares new model
    accuracy vs old and rolls back if the new model is worse.
    """

    def __init__(self, metric_registry=None):
        self._last_n_entries: int = 0
        self._metric_registry = metric_registry

    def check_and_retrain(
        self,
        journal_path: str,
        current_model_path: str,
        min_new_entries: int = 20,
        epochs: int = 100,
    ) -> tuple[bool, dict]:
        """Check if retraining is needed and retrain if so.

        Retraining triggers when the journal has grown by at least
        min_new_entries since the last retrain.

        Args:
            journal_path: Path to hypothesis_journal.jsonl.
            current_model_path: Path to the current surrogate model weights.
            min_new_entries: Minimum new entries to trigger retraining.
            epochs: Training epochs for retraining.

        Returns:
            (retrained, metrics) — whether retraining occurred and the
            training/evaluation metrics.
        """
        # Extract current training data
        extractor = JournalDataExtractor()
        enricher = FeatureEnricher()
        examples = extractor.extract(
            journal_path, enricher=enricher, metric_registry=self._metric_registry,
        )

        n_entries = len(examples)
        new_entries = n_entries - self._last_n_entries

        if new_entries < min_new_entries:
            logger.info(
                "Only %d new entries (%d total), need %d to retrain",
                new_entries, n_entries, min_new_entries,
            )
            return False, {"n_entries": n_entries, "new_entries": new_entries}

        if n_entries < 10:
            logger.info("Not enough total examples (%d) to train", n_entries)
            return False, {"n_entries": n_entries, "new_entries": new_entries}

        logger.info(
            "Retraining triggered: %d new entries (%d total)",
            new_entries, n_entries,
        )

        # Backup current model
        backup_path = current_model_path + ".backup"
        has_existing = os.path.exists(current_model_path)
        if has_existing:
            shutil.copy2(current_model_path, backup_path)
            logger.info("Backed up current model to %s", backup_path)

        # Train new model
        input_dim = len(examples[0].feature_vector) if examples else 279
        trainer = SurrogateTrainer(input_dim=input_dim)

        try:
            train_metrics = trainer.train(examples, epochs=epochs)
        except Exception as exc:
            logger.error("Retraining failed: %s", exc)
            if has_existing and os.path.exists(backup_path):
                self.rollback(backup_path, current_model_path)
            return False, {"error": str(exc)}

        # Evaluate new model vs old
        if has_existing:
            should_keep = self._compare_models(
                trainer, current_model_path, examples,
            )
            if not should_keep:
                logger.warning("New model is worse — rolling back")
                self.rollback(backup_path, current_model_path)
                return False, {
                    "rolled_back": True,
                    "train_metrics": train_metrics,
                }

        # Save new model
        trainer.save(current_model_path)
        self._last_n_entries = n_entries

        logger.info("Retraining complete, new model saved")
        return True, train_metrics

    @staticmethod
    def rollback(backup_path: str, model_path: str):
        """Restore a backed-up model.

        Args:
            backup_path: Path to the backup file.
            model_path: Path where the model should be restored.
        """
        if not os.path.exists(backup_path):
            logger.error("Backup not found at %s — cannot rollback", backup_path)
            return

        shutil.copy2(backup_path, model_path)
        logger.info("Rolled back model from %s", backup_path)

    @staticmethod
    def _compare_models(
        new_trainer: SurrogateTrainer,
        old_model_path: str,
        examples: list,
    ) -> bool:
        """Compare new model vs old on the training data.

        Returns True if the new model is at least as good.
        """
        if not examples:
            return True

        # Get new model predictions
        new_preds = []
        actuals = []
        for ex in examples:
            pred = new_trainer.predict(ex.feature_vector)
            new_preds.append(pred.predicted_delta)
            actuals.append(ex.actual_delta)

        # Load and evaluate old model
        try:
            input_dim = len(examples[0].feature_vector)
            old_trainer = SurrogateTrainer(input_dim=input_dim)
            old_trainer.load(old_model_path)

            old_preds = []
            for ex in examples:
                pred = old_trainer.predict(ex.feature_vector)
                old_preds.append(pred.predicted_delta)
        except Exception as exc:
            logger.warning("Could not load old model for comparison: %s", exc)
            return True  # Keep new model if old can't be loaded

        evaluator = SurrogateEvaluator()
        new_eval = evaluator.evaluate(new_preds, actuals)
        old_eval = evaluator.evaluate(old_preds, actuals)

        new_mae = new_eval.get("mae", float("inf"))
        old_mae = old_eval.get("mae", float("inf"))

        logger.info(
            "Model comparison — new MAE: %.6f, old MAE: %.6f",
            new_mae, old_mae,
        )

        # Allow a small tolerance: new model can be slightly worse
        return new_mae <= old_mae * 1.05
