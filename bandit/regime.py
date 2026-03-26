"""
Regime transition manager for the Adaptive Bandit pipeline.

Manages transitions between no_bandit, conservative_bandit, and full_bandit
regimes based on journal entry counts and performance windows.
"""

import json

from bandit.schemas import BanditState, ArmState


# Transition thresholds
_THRESHOLD_CONSERVATIVE = 30   # no_bandit -> conservative_bandit
_THRESHOLD_FULL = 100          # conservative_bandit -> full_bandit
_DOWNGRADE_WINDOW = 50         # iterations to evaluate for downgrade


class RegimeTransitionManager:
    """Check and recommend regime transitions based on evidence count
    and rolling performance."""

    def check_transition(
        self, state: BanditState, journal_path: str,
    ) -> tuple[str, str | None]:
        """Return ``(current_regime, new_regime_or_None)``.

        Upgrade rules:
          - ``no_bandit`` -> ``conservative_bandit`` at 30 entries.
          - ``conservative_bandit`` -> ``full_bandit`` at 100 entries.

        Downgrade recommendation:
          - If the bandit underperforms over the last 50-iteration window
            (success rate drops below 50% of the all-time rate), recommend
            downgrading one level.
        """
        current = state.regime
        total = state.global_iteration

        # --- Upgrades ---
        if current == "no_bandit" and total >= _THRESHOLD_CONSERVATIVE:
            return current, "conservative_bandit"

        if current == "conservative_bandit" and total >= _THRESHOLD_FULL:
            return current, "full_bandit"

        # --- Downgrade check (only in active bandit modes) ---
        if current in ("conservative_bandit", "full_bandit"):
            new_regime = self._check_downgrade(state, journal_path, current)
            if new_regime is not None:
                return current, new_regime

        return current, None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_downgrade(
        self, state: BanditState, journal_path: str, current: str,
    ) -> str | None:
        """Evaluate rolling performance and recommend downgrade if warranted."""
        try:
            with open(journal_path) as f:
                entries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        if not isinstance(entries, list) or len(entries) < _DOWNGRADE_WINDOW:
            return None

        def _success(e: dict) -> bool:
            return (e.get("verdict") or "").lower() in (
                "accepted", "improved", "improvement")

        # All-time rate
        total_success = sum(1 for e in entries if _success(e))
        total_count = len(entries)
        alltime_rate = total_success / total_count if total_count > 0 else 0.0

        # Rolling window rate
        recent = entries[-_DOWNGRADE_WINDOW:]
        recent_success = sum(1 for e in recent if _success(e))
        recent_rate = recent_success / len(recent)

        # Recommend downgrade if rolling rate < 50% of all-time rate
        if alltime_rate > 0 and recent_rate < 0.5 * alltime_rate:
            if current == "full_bandit":
                return "conservative_bandit"
            if current == "conservative_bandit":
                return "no_bandit"

        return None
