"""
SurrogateMetricProposer — proposes new diagnostic metrics derived from
surrogate triage patterns to the Model Scientist's CriticAgent.
"""

import time
from model_scientist.schemas import MetricDefinition


class SurrogateMetricProposer:
    """Proposes metrics from surrogate triage data to the Model Scientist metric evolution system."""

    # Built-in proposals derived from surrogate triage patterns
    _PROPOSALS = [
        {
            "trigger": lambda stats: stats.get("n_paper_evaluations", 0) >= 10,
            "name": "surrogate_prediction_confidence",
            "description": (
                "How confident the surrogate is about the current model state. "
                "Low confidence indicates the model has entered an unexplored regime."
            ),
            "computation_method": (
                "# Uses surrogate prediction variance across recent candidates\n"
                "scores = diagnostics.get('recent_surrogate_scores', [])\n"
                "if len(scores) < 2:\n"
                "    result = 0.0\n"
                "else:\n"
                "    mean = sum(scores) / len(scores)\n"
                "    var = sum((s - mean) ** 2 for s in scores) / len(scores)\n"
                "    result = 1.0 / (1.0 + var ** 0.5)\n"
            ),
            "rationale": (
                "When the surrogate's predictions cluster tightly, it's confident about "
                "the current state. When they're spread out, the model is in unfamiliar territory."
            ),
        },
        {
            "trigger": lambda stats: stats.get("n_paper_evaluations", 0) >= 20,
            "name": "paper_technique_novelty_score",
            "description": (
                "Embedding distance of the best paper-sourced diff from all historically "
                "attempted diffs. High novelty means the pipeline is surfacing genuinely new ideas."
            ),
            "computation_method": (
                "novelty = diagnostics.get('avg_paper_novelty', 0.0)\n"
                "result = novelty\n"
            ),
            "rationale": (
                "Tracks whether the paper pipeline is breaking the LLM knowledge ceiling "
                "by introducing techniques far from the internal proposal distribution."
            ),
        },
        {
            "trigger": lambda stats: stats.get("n_paper_evaluations", 0) >= 30,
            "name": "surrogate_calibration_drift",
            "description": (
                "Running MAE of surrogate predictions — rising drift indicates the surrogate "
                "needs retraining or the modification landscape has shifted."
            ),
            "computation_method": (
                "errors = diagnostics.get('recent_prediction_errors', [])\n"
                "if not errors:\n"
                "    result = 0.0\n"
                "else:\n"
                "    result = sum(abs(e) for e in errors) / len(errors)\n"
            ),
            "rationale": (
                "A well-calibrated surrogate is the backbone of efficient triage. "
                "Rising drift means the surrogate is losing touch with the current model."
            ),
        },
        {
            "trigger": lambda stats: stats.get("total_papers_ingested", 0) >= 50,
            "name": "ingestion_precision",
            "description": (
                "Fraction of ingested papers whose techniques led to accepted modifications. "
                "Measures how well the relevance filter targets productive papers."
            ),
            "computation_method": (
                "accepted = diagnostics.get('paper_accepted_count', 0)\n"
                "total = diagnostics.get('paper_evaluated_count', 1)\n"
                "result = accepted / max(total, 1)\n"
            ),
            "rationale": (
                "If ingestion precision is low, the relevance filter is wasting GPU time "
                "on unpromising papers. If high, the filter is well-calibrated."
            ),
        },
    ]

    def propose_metrics(self, triage_stats: dict, existing_metrics: list = None) -> list:
        """Propose new metrics from surrogate triage patterns.

        Args:
            triage_stats: Dict with keys like n_paper_evaluations, total_papers_ingested, etc.
            existing_metrics: List of existing MetricDefinition to avoid duplicates.

        Returns:
            List of MetricDefinition proposals.
        """
        existing_names = set()
        if existing_metrics:
            for m in existing_metrics:
                name = m.name if isinstance(m, MetricDefinition) else m.get("name", "")
                existing_names.add(name)

        proposals = []
        for template in self._PROPOSALS:
            if template["name"] in existing_names:
                continue
            try:
                if template["trigger"](triage_stats):
                    metric = MetricDefinition(
                        name=template["name"],
                        description=template["description"],
                        computation_method=template["computation_method"],
                        rationale=template["rationale"],
                        source="surrogate_triage",
                        created_at=time.time(),
                        status="candidate",
                    )
                    proposals.append(metric)
            except Exception:
                continue

        return proposals
