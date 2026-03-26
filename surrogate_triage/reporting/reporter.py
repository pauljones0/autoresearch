"""
PipelineReporter — generates human-readable weekly reports consolidating
data from both the Model Scientist Pipeline and the Surrogate Triage Pipeline.
"""

import json
import os
import time
from datetime import datetime

from ..schemas import load_jsonl


class PipelineReporter:
    """Generates weekly reports for the full pipeline."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
        )

    def generate_weekly_report(
        self,
        journal_reader=None,
        source_tracker=None,
        extraction_tracker=None,
        queue_manager=None,
        ceiling_monitor=None,
        meta_monitor=None,
    ) -> str:
        """Generate a comprehensive weekly report.

        Returns:
            Markdown-formatted report string.
        """
        now = datetime.now()
        week_start = now.strftime("%Y-%m-%d")

        lines = []
        lines.append(f"# Weekly Pipeline Report — {week_start}")
        lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # --- Section 1: Paper Ingestion ---
        lines.append("## Paper Ingestion")
        papers_index = load_jsonl(os.path.join(self.data_dir, "papers_index.jsonl"))
        week_seconds = 7 * 86400
        cutoff = time.time() - week_seconds
        recent_papers = [p for p in papers_index if p.get("fetched_at", 0) > cutoff]
        filtered_out = [p for p in recent_papers if p.get("filtered_out")]
        lines.append(f"- Papers fetched this week: {len(recent_papers)}")
        lines.append(f"- Papers filtered out: {len(filtered_out)}")
        lines.append(f"- Papers passed filter: {len(recent_papers) - len(filtered_out)}")
        lines.append("")

        # --- Section 2: Technique Extraction ---
        lines.append("## Technique Extraction")
        techniques = load_jsonl(os.path.join(self.data_dir, "technique_descriptions.jsonl"))
        recent_techniques = [t for t in techniques if t.get("extracted_at", 0) > cutoff]
        deduped = [t for t in recent_techniques if t.get("deduplicated")]
        already_explored = [t for t in recent_techniques if t.get("already_explored")]
        lines.append(f"- Techniques extracted: {len(recent_techniques)}")
        lines.append(f"- Deduplicated: {len(deduped)}")
        lines.append(f"- Already explored: {len(already_explored)}")
        lines.append("")

        # --- Section 3: Constraint Pre-Filter ---
        lines.append("## Constraint Pre-Filter")
        diffs = load_jsonl(os.path.join(self.data_dir, "synthetic_diffs.jsonl"))
        recent_diffs = [d for d in diffs if d.get("generated_at", 0) > cutoff]
        constrained = [d for d in recent_diffs if d.get("constraint_penalty", 0) > 0]
        lines.append(f"- Diffs generated: {len(recent_diffs)}")
        lines.append(f"- Diffs with constraint matches: {len(constrained)}")
        if constrained:
            avg_penalty = sum(d["constraint_penalty"] for d in constrained) / len(constrained)
            lines.append(f"- Average constraint penalty: {avg_penalty:.4f}")
        lines.append("")

        # --- Section 4: Surrogate Scoring ---
        lines.append("## Surrogate Scoring")
        if queue_manager:
            try:
                queue = queue_manager.get_all()
                lines.append(f"- Current queue size: {len(queue)}")
                if queue:
                    scores = [e.get("adjusted_score", 0) if isinstance(e, dict)
                              else getattr(e, "adjusted_score", 0) for e in queue]
                    lines.append(f"- Best score in queue: {min(scores):.6f}")
                    lines.append(f"- Worst score in queue: {max(scores):.6f}")
            except Exception:
                lines.append("- Queue: unavailable")
        lines.append("")

        # --- Section 5: GPU Evaluations ---
        lines.append("## GPU Evaluations")
        if journal_reader:
            try:
                journal_reader.reload()
                all_entries = journal_reader._entries if hasattr(journal_reader, "_entries") else []
                recent_entries = [e for e in all_entries if e.get("timestamp", 0) > cutoff]
                paper_entries = [e for e in recent_entries if "source:paper" in e.get("tags", [])]
                internal_entries = [e for e in recent_entries if "source:paper" not in e.get("tags", [])]

                paper_accepted = sum(1 for e in paper_entries if e.get("verdict") == "accepted")
                internal_accepted = sum(1 for e in internal_entries if e.get("verdict") == "accepted")

                lines.append(f"- Total evaluations: {len(recent_entries)}")
                lines.append(f"- Paper-sourced: {len(paper_entries)} ({paper_accepted} accepted)")
                lines.append(f"- Internal: {len(internal_entries)} ({internal_accepted} accepted)")
                if paper_entries:
                    paper_rate = paper_accepted / len(paper_entries)
                    lines.append(f"- Paper acceptance rate: {paper_rate:.1%}")
                if internal_entries:
                    internal_rate = internal_accepted / len(internal_entries)
                    lines.append(f"- Internal acceptance rate: {internal_rate:.1%}")
            except Exception:
                lines.append("- Journal: unavailable")
        lines.append("")

        # --- Section 6: Source Quality ---
        lines.append("## Source Quality Updates")
        if source_tracker:
            try:
                for dim in ["author", "category", "technique_category"]:
                    top = source_tracker.get_top_sources(dim, n=3)
                    if top:
                        lines.append(f"- Top {dim}s:")
                        for s in top:
                            name = s.get("value", "?") if isinstance(s, dict) else getattr(s, "value", "?")
                            rate = s.get("success_rate", 0) if isinstance(s, dict) else getattr(s, "success_rate", 0)
                            n_eval = s.get("total_evaluated", 0) if isinstance(s, dict) else getattr(s, "total_evaluated", 0)
                            lines.append(f"    - {name}: {rate:.0%} success ({n_eval} evaluated)")
            except Exception:
                lines.append("- Source quality: unavailable")
        lines.append("")

        # --- Section 7: Knowledge Ceiling ---
        lines.append("## Knowledge Ceiling")
        if ceiling_monitor:
            try:
                report = ceiling_monitor.get_report()
                lines.append(report)
            except Exception:
                lines.append("- Ceiling monitor: unavailable")
        lines.append("")

        # --- Section 8: Meta-Learning ---
        lines.append("## Meta-Learning Health")
        if meta_monitor:
            try:
                report = meta_monitor.generate_report()
                # Indent under this section
                for line in report.split("\n"):
                    if line.startswith("#"):
                        continue  # skip duplicate headers
                    lines.append(line)
            except Exception:
                lines.append("- Meta-learning monitor: unavailable")
        lines.append("")

        report_text = "\n".join(lines)

        # Save to file
        report_path = os.path.join(self.data_dir, "weekly_report.md")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_text)

        return report_text
