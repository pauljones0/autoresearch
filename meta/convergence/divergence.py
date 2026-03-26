"""Divergence detection for production IR regression."""

import copy

from meta.schemas import MetaBanditState, DivergenceAlert


class DivergenceWatcher:
    """Watches for significant drops in production improvement rate."""

    def __init__(self, consecutive_threshold: int = 3):
        self.consecutive_threshold = consecutive_threshold

    def check(
        self,
        meta_state: MetaBanditState,
        recent_ir_windows: list,
        baseline_ir: float,
    ) -> "DivergenceAlert | None":
        """Check if rolling IR has dropped below baseline - 2 sigma.

        Args:
            meta_state: Current meta-bandit state.
            recent_ir_windows: List of dicts with at least 'mean_ir' key,
                ordered oldest-first.
            baseline_ir: Baseline aggregate IR dict or float.  If dict,
                expects 'mean_ir' and 'std_ir' keys.

        Returns:
            DivergenceAlert if divergence detected, else None.
        """
        if isinstance(baseline_ir, dict):
            bl_mean = baseline_ir.get("mean_ir", 0.0)
            bl_std = baseline_ir.get("std_ir", 0.0)
        elif hasattr(baseline_ir, "mean_ir"):
            bl_mean = baseline_ir.mean_ir
            bl_std = getattr(baseline_ir, "std_ir", 0.0)
        else:
            bl_mean = float(baseline_ir)
            bl_std = self._estimate_std(recent_ir_windows)

        threshold = bl_mean - 2.0 * bl_std

        # Count consecutive windows below threshold (from most recent)
        consecutive = 0
        for w in reversed(recent_ir_windows):
            ir = w.get("mean_ir", w) if isinstance(w, dict) else float(w)
            if ir < threshold:
                consecutive += 1
            else:
                break

        if consecutive < self.consecutive_threshold:
            return None

        # Divergence detected
        most_recent_ir = (
            recent_ir_windows[-1].get("mean_ir", 0.0)
            if isinstance(recent_ir_windows[-1], dict)
            else float(recent_ir_windows[-1])
        )
        drop = bl_mean - most_recent_ir

        # Soft-reset posteriors
        self._soft_reset_posteriors(meta_state)

        return DivergenceAlert(
            triggered=True,
            current_ir=most_recent_ir,
            baseline_ir=bl_mean,
            drop_magnitude=drop,
            windows_below_threshold=consecutive,
            recommendation="re-enter active mode: IR has dropped significantly below baseline",
        )

    def _soft_reset_posteriors(self, meta_state: MetaBanditState) -> None:
        """Multiply all alpha/beta by 0.5 to soften prior beliefs."""
        for dim in meta_state.dimensions.values():
            if not hasattr(dim, "variant_posteriors"):
                continue
            for var_key, post in dim.variant_posteriors.items():
                if isinstance(post, dict):
                    post["alpha"] = max(1.0, post.get("alpha", 1.0) * 0.5)
                    post["beta"] = max(1.0, post.get("beta", 1.0) * 0.5)

    def _estimate_std(self, windows: list) -> float:
        """Estimate std from window IRs when not provided."""
        if not windows:
            return 0.0
        irs = []
        for w in windows:
            ir = w.get("mean_ir", w) if isinstance(w, dict) else float(w)
            irs.append(ir)
        if len(irs) < 2:
            return 0.0
        mean = sum(irs) / len(irs)
        var = sum((x - mean) ** 2 for x in irs) / (len(irs) - 1)
        return var ** 0.5
