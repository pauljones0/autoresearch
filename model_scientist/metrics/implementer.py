"""
Phase 4: MetricImplementer — compiles and executes metric computation code
in a sandboxed environment, validates outputs.
"""

import math
from ..schemas import DiagnosticsReport, MetricDefinition


# Restricted builtins allowed inside metric code
_SAFE_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


class MetricImplementer:
    """Compiles metric code strings and executes them against diagnostics data."""

    def compute_metric(
        self,
        metric: MetricDefinition,
        diagnostics: DiagnosticsReport,
    ) -> float:
        """Execute a metric's computation_method against a single diagnostics report.

        Returns:
            A finite float value.

        Raises:
            ValueError: If execution fails or produces invalid output.
        """
        diag_dict = diagnostics.to_dict() if isinstance(diagnostics, DiagnosticsReport) else diagnostics
        code = metric.computation_method if isinstance(metric, MetricDefinition) else metric.get("computation_method", "")

        if not code.strip():
            raise ValueError("Empty computation_method")

        namespace = {"__builtins__": _SAFE_BUILTINS, "diagnostics": diag_dict, "math": math}

        try:
            compiled = compile(code, f"<metric:{metric.name if isinstance(metric, MetricDefinition) else metric.get('name', '?')}>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Syntax error in metric code: {exc}") from exc

        try:
            exec(compiled, namespace)  # noqa: S102
        except Exception as exc:
            raise ValueError(f"Runtime error in metric code: {exc}") from exc

        result = namespace.get("result")
        if result is None:
            raise ValueError("Metric code did not assign a 'result' variable")

        try:
            value = float(result)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric result is not a number: {result!r}") from exc

        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Metric result is {value}")

        return value

    def implement(
        self,
        metric: MetricDefinition,
        test_data: list,
    ) -> tuple:
        """Validate a metric against a list of DiagnosticsReports.

        Returns:
            (success, error_or_none, values) where values is a list of floats.
        """
        if not test_data:
            return False, "No test data provided", []

        values: list[float] = []
        for report in test_data:
            try:
                val = self.compute_metric(metric, report)
                values.append(val)
            except ValueError as exc:
                return False, str(exc), values

        # Check for non-trivial variance (all identical values are not useful)
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            if variance == 0.0:
                return False, "Metric produces constant values across all test data", values

        return True, None, values
