"""
Variance-cost analyzer for evaluation protocols.
"""

from meta.schemas import VarianceCostReport


class MetaVarianceCostAnalyzer:
    """Cost-effectiveness analysis of evaluation protocols."""

    def analyze(self, protocol_results: list) -> VarianceCostReport:
        """Analyze variance vs cost for each protocol.

        protocol_results: list of dicts with keys:
            protocol_id, variance, time_seconds, mdes
        """
        per_protocol = {}
        best_id = ""
        best_ce = -1.0

        for pr in protocol_results:
            pid = pr.get("protocol_id", "unknown")
            variance = pr.get("variance", 1.0)
            time_s = pr.get("time_seconds", 1.0)
            mdes = pr.get("mdes", 1.0)
            ce = 1.0 / max(1e-10, variance * time_s)
            per_protocol[pid] = {
                "variance": variance,
                "time_seconds": time_s,
                "cost_effectiveness": ce,
                "mdes": mdes,
            }
            if ce > best_ce:
                best_ce = ce
                best_id = pid

        # Check for two-stage recommendation
        two_stage = any(
            pr.get("protocol_id", "").startswith("two_stage")
            and pr.get("mdes", 1.0) < 0.1
            for pr in protocol_results
        )

        return VarianceCostReport(
            per_protocol=per_protocol,
            recommended_protocol_id=best_id,
            recommended_two_stage=two_stage,
        )
