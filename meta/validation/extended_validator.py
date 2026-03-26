"""
Extended validation for the meta-optimization loop.

Runs N iterations verifying no boundary violations, state corruptions,
eval hash changes, budget overruns, or health audit failures.
"""

import hashlib
import json
import random
import time

from meta.schemas import (
    MetaBanditState,
    DimensionState,
    ExtendedMetaValidationResult,
    MetaExperimentResult,
)


class MetaExtendedValidator:
    """Run extended validation of the meta-optimization loop.

    Simulates N iterations of meta-experiment selection, execution, and
    posterior update to verify invariants hold throughout.
    """

    def __init__(self, meta_state: MetaBanditState, meta_parameters: list = None,
                 seed: int = 42):
        """
        Args:
            meta_state: Current meta-bandit state.
            meta_parameters: List of MetaParameter definitions (for range checks).
            seed: RNG seed for reproducibility.
        """
        self._state = meta_state
        self._params = meta_parameters or []
        self._seed = seed
        self._param_ranges = {}
        for p in self._params:
            self._param_ranges[p.param_id] = p.valid_range

    def validate(self, n_iterations: int = 500) -> ExtendedMetaValidationResult:
        """Run extended validation over n_iterations simulated meta-steps.

        Checks:
            1. No boundary violations (values stay within valid_range).
            2. No state corruptions (dimensions remain well-formed).
            3. Eval hash unchanged (evaluation identity is stable).
            4. Budget within bounds (budget_used <= budget_fraction * cycle_length).
            5. Periodic health audits pass (every 50 iterations).

        Returns:
            ExtendedMetaValidationResult with counts of each failure type.
        """
        rng = random.Random(self._seed)
        boundary_violations = 0
        state_corruptions = 0
        eval_hash_changes = 0
        budget_deviation_pct = 0.0
        health_failures = 0
        meta_experiments_run = 0
        promotions = 0

        # Snapshot the initial eval hash
        eval_hash = self._compute_eval_hash(self._state)

        for i in range(n_iterations):
            # --- 1. State integrity check ---
            issues = self._state.validate()
            if issues:
                state_corruptions += len(issues)

            # --- 2. Boundary violation check ---
            for param_id, dim in self._state.dimensions.items():
                if not isinstance(dim, DimensionState):
                    state_corruptions += 1
                    continue
                violation = self._check_boundary(param_id, dim.current_best)
                if violation:
                    boundary_violations += 1

            # --- 3. Eval hash stability ---
            current_hash = self._compute_eval_hash(self._state)
            if current_hash != eval_hash:
                eval_hash_changes += 1

            # --- 4. Simulate a meta-experiment step ---
            if self._should_run_meta_experiment(self._state, rng):
                meta_experiments_run += 1
                self._state.total_meta_experiments += 1
                self._state.budget_used += 1

                # Simulate a possible promotion
                if rng.random() < 0.1:
                    promotions += 1
                    self._simulate_promotion(rng)

            # Advance iteration
            self._state.global_meta_iteration += 1

            # --- 5. Periodic health audit ---
            if (i + 1) % 50 == 0:
                if not self._health_audit():
                    health_failures += 1

        # Budget deviation
        max_budget = self._state.budget_fraction * self._state.budget_cycle_length
        if max_budget > 0:
            budget_deviation_pct = abs(
                self._state.budget_used - max_budget
            ) / max_budget * 100.0
        else:
            budget_deviation_pct = 0.0 if self._state.budget_used == 0 else 100.0

        passed = (
            boundary_violations == 0
            and state_corruptions == 0
            and eval_hash_changes == 0
            and budget_deviation_pct <= 10.0
            and health_failures == 0
        )

        return ExtendedMetaValidationResult(
            passed=passed,
            n_iterations=n_iterations,
            boundary_violations=boundary_violations,
            state_corruptions=state_corruptions,
            eval_hash_changes=eval_hash_changes,
            budget_deviation_percent=round(budget_deviation_pct, 2),
            health_audit_failures=health_failures,
            meta_experiments_run=meta_experiments_run,
            promotions=promotions,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_eval_hash(self, state: MetaBanditState) -> str:
        """Hash the evaluation-relevant parts of state (should never change)."""
        # The eval identity is the schema version + dimension param_ids.
        # In a real system this would hash the evaluation code/dataset.
        identity = {
            "schema_version": state.metadata.get("schema_version", ""),
            "dimensions": sorted(state.dimensions.keys()),
        }
        raw = json.dumps(identity, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def _check_boundary(self, param_id: str, value) -> bool:
        """Return True if value violates the parameter's valid range."""
        vr = self._param_ranges.get(param_id, {})
        if not vr or value is None:
            return False
        if "enum" in vr:
            return value not in vr["enum"]
        lo = vr.get("min")
        hi = vr.get("max")
        try:
            v = float(value)
        except (TypeError, ValueError):
            return True
        if lo is not None and v < float(lo):
            return True
        if hi is not None and v > float(hi):
            return True
        return False

    def _should_run_meta_experiment(self, state: MetaBanditState,
                                     rng: random.Random) -> bool:
        """Decide if a meta-experiment fires this iteration."""
        max_budget = state.budget_fraction * state.budget_cycle_length
        if state.budget_used >= max_budget:
            return False
        # Probability proportional to remaining budget
        remaining = max_budget - state.budget_used
        prob = min(remaining / max(max_budget, 1), 0.3)
        return rng.random() < prob

    def _simulate_promotion(self, rng: random.Random):
        """Simulate promoting a variant in a random dimension."""
        if not self._state.dimensions:
            return
        dim_ids = list(self._state.dimensions.keys())
        dim_id = rng.choice(dim_ids)
        dim = self._state.dimensions[dim_id]
        if isinstance(dim, DimensionState) and dim.variants:
            dim.current_best = rng.choice(dim.variants)
            dim.last_promoted = time.time()

    def _health_audit(self) -> bool:
        """Run a periodic health audit on the meta-state."""
        # Check basic invariants
        if self._state.meta_regime not in ("baseline", "active", "maintenance"):
            return False
        if self._state.budget_fraction < 0 or self._state.budget_fraction > 1:
            return False
        if self._state.global_meta_iteration < 0:
            return False
        for dim_id, dim in self._state.dimensions.items():
            if not isinstance(dim, DimensionState):
                return False
            for var_key, post in dim.variant_posteriors.items():
                if isinstance(post, dict):
                    if post.get("alpha", 1) < 1 or post.get("beta", 1) < 1:
                        return False
        return True
