"""Maintenance mode management for the meta-loop."""

import time
import copy

from meta.schemas import MetaBanditState


class MaintenanceModeManager:
    """Manages transition to and behaviour within maintenance mode."""

    def enter_maintenance(self, meta_state: MetaBanditState, meta_config: dict) -> MetaBanditState:
        """Transition meta-state into maintenance mode.

        Actions:
        - Set regime to 'maintenance'.
        - Reduce budget fraction to 5%.
        - Freeze the current best config.
        - Log the transition in metadata.
        """
        new_state = copy.deepcopy(meta_state)
        new_state.meta_regime = "maintenance"
        new_state.budget_fraction = 0.05
        # Freeze best config
        new_state.best_config = dict(new_state.best_config)
        new_state.metadata["entered_maintenance_at"] = time.time()
        new_state.metadata["maintenance_entry_iteration"] = new_state.global_meta_iteration
        new_state.metadata["pre_maintenance_budget"] = meta_state.budget_fraction
        new_state.metadata["last_updated"] = time.time()
        return new_state

    def should_promote_in_maintenance(self, p_value: float) -> bool:
        """In maintenance mode, require p < 0.01 for promotions."""
        return p_value < 0.01
