"""
Discretize continuous and integer meta-parameters into finite variant sets.

For continuous params: evenly spaced values including the default.
For T_base (temperature): log-spaced values.
For int params: quantile-based selection.
For bool params: [True, False].
"""

import math

from meta.schemas import MetaParameter


class MetaVariantDiscretizer:
    """Convert continuous/int/bool meta-parameters into discrete variant lists."""

    def discretize(self, param: MetaParameter, n_variants: int = 5) -> list:
        """Discretize a single parameter into a list of variant values.

        Args:
            param: The meta-parameter definition.
            n_variants: Target number of variants for continuous/int types.

        Returns:
            List of discrete variant values.
        """
        if param.type == "bool":
            return [True, False]

        if param.type == "str":
            enum_vals = param.valid_range.get("enum", [])
            return enum_vals if enum_vals else [param.default_value]

        lo = param.valid_range.get("min")
        hi = param.valid_range.get("max")
        if lo is None or hi is None:
            return [param.default_value] if param.default_value is not None else []

        if param.type == "int":
            return self._discretize_int(lo, hi, param.default_value, n_variants)

        # float — check if log-spacing is appropriate (temperature-like)
        if param.category == "temperature" or "T_base" in param.param_id:
            return self._discretize_log(lo, hi, param.default_value, n_variants)

        return self._discretize_linear(lo, hi, param.default_value, n_variants)

    def discretize_all(self, params: list) -> dict:
        """Discretize all parameters.

        Args:
            params: List of MetaParameter objects.

        Returns:
            Dict mapping param_id -> list of variant values.
        """
        return {p.param_id: self.discretize(p) for p in params}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discretize_linear(self, lo: float, hi: float,
                           default, n: int) -> list:
        """Evenly spaced values including the default."""
        if n <= 1:
            return [default if default is not None else lo]

        step = (hi - lo) / (n - 1)
        values = [lo + i * step for i in range(n)]

        # Ensure default is included
        if default is not None and lo <= default <= hi:
            values = self._insert_default(values, float(default))

        return sorted(set(round(v, 10) for v in values))

    def _discretize_log(self, lo: float, hi: float,
                        default, n: int) -> list:
        """Log-spaced values for temperature-like parameters."""
        if lo <= 0:
            lo = 1e-6
        if hi <= 0:
            hi = 1e-6

        log_lo = math.log(lo)
        log_hi = math.log(hi)

        if n <= 1:
            return [default if default is not None else lo]

        step = (log_hi - log_lo) / (n - 1)
        values = [math.exp(log_lo + i * step) for i in range(n)]

        if default is not None and lo <= default <= hi:
            values = self._insert_default(values, float(default))

        return sorted(set(round(v, 10) for v in values))

    def _discretize_int(self, lo: int, hi: int,
                        default, n: int) -> list:
        """Quantile-based integer selection."""
        span = hi - lo + 1
        if span <= n:
            values = list(range(int(lo), int(hi) + 1))
        else:
            values = []
            for i in range(n):
                frac = i / (n - 1) if n > 1 else 0
                values.append(int(round(lo + frac * (hi - lo))))

        if default is not None and int(lo) <= int(default) <= int(hi):
            val = int(default)
            if val not in values:
                values.append(val)

        return sorted(set(values))

    def _insert_default(self, values: list, default: float) -> list:
        """Insert default into values, replacing the closest existing value."""
        if default in values:
            return values

        # Find closest value and replace it
        closest_idx = min(range(len(values)),
                          key=lambda i: abs(values[i] - default))
        values[closest_idx] = default
        return values
