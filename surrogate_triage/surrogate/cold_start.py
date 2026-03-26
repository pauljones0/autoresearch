"""
Phase 2 — ColdStartManager: manage surrogate filtering regimes
based on the number of journal entries available for training.
"""

import logging

logger = logging.getLogger(__name__)

# Regime thresholds
_REGIME_NO_FILTER = "no_filter"
_REGIME_CONSERVATIVE = "conservative"
_REGIME_FULL = "full"

_THRESHOLDS = {
    _REGIME_NO_FILTER: {"min": 0, "max": 49, "filter_fraction": 1.0, "use_surrogate": False},
    _REGIME_CONSERVATIVE: {"min": 50, "max": 199, "filter_fraction": 0.5, "use_surrogate": True},
    _REGIME_FULL: {"min": 200, "max": float("inf"), "filter_fraction": 0.2, "use_surrogate": True},
}


class ColdStartManager:
    """Manages cold-start regimes for surrogate-based filtering.

    Three regimes based on journal size:
        <50 entries:   no surrogate filtering, all candidates evaluated
        50-200 entries: conservative threshold (top 50%)
        200+ entries:  full threshold (top 20%)
    """

    def __init__(self):
        self._last_regime: str | None = None

    def get_regime(self, n_entries: int) -> tuple[str, float, bool]:
        """Determine the filtering regime for the given journal size.

        Args:
            n_entries: Number of entries in the hypothesis journal.

        Returns:
            (regime_name, filter_fraction, use_surrogate)
        """
        if n_entries < 0:
            n_entries = 0

        if n_entries < 50:
            regime = _REGIME_NO_FILTER
        elif n_entries < 200:
            regime = _REGIME_CONSERVATIVE
        else:
            regime = _REGIME_FULL

        cfg = _THRESHOLDS[regime]

        # Log regime transitions
        if self._last_regime is not None and self._last_regime != regime:
            logger.info(
                "Cold-start regime transition: %s -> %s (n_entries=%d)",
                self._last_regime, regime, n_entries,
            )
        self._last_regime = regime

        return regime, cfg["filter_fraction"], cfg["use_surrogate"]

    def should_filter(self, n_entries: int) -> bool:
        """Return True if surrogate filtering should be applied.

        Args:
            n_entries: Number of entries in the hypothesis journal.
        """
        _, _, use_surrogate = self.get_regime(n_entries)
        return use_surrogate
