"""
Phase 3 — AblationJournalWriter: Extend journal entries with ablation data
and feed stripped-component patterns into the failure mining pipeline.
"""

import json
import os
import time

from ..schemas import AblationReport, AblationResult


# Default path for stripped-component pattern data
DEFAULT_PATTERN_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'ablation_patterns.jsonl')


class AblationJournalWriter:
    """Record ablation results in journal entries and emit pattern data."""

    def __init__(self, pattern_output_path: str = None):
        """
        Args:
            pattern_output_path: Path to write stripped-component pattern data.
                Defaults to data/ablation_patterns.jsonl.
        """
        self.pattern_output_path = pattern_output_path or DEFAULT_PATTERN_PATH

    def record_ablation(self, journal_writer, entry_id: str,
                        ablation_report: AblationReport):
        """Extend a journal entry with ablation data.

        Args:
            journal_writer: Object with update_entry(id, **kwargs) or
                a dict-like journal store. If it has an `update_entry` method,
                that is called. Otherwise, the report is stored in
                journal_writer[entry_id].
            entry_id: The journal entry ID to update.
            ablation_report: Completed ablation report.
        """
        ablation_data = self._format_ablation_data(ablation_report)
        components_summary = self._format_components(ablation_report)
        stripped_ids = ablation_report.stripped_components

        update_fields = {
            "ablation_data": ablation_data,
            "components": components_summary,
            "stripped_components": stripped_ids,
            "final_val_bpb": ablation_report.final_val_bpb,
        }

        # Update journal entry
        if hasattr(journal_writer, 'update_entry'):
            journal_writer.update_entry(entry_id, **update_fields)
        elif hasattr(journal_writer, '__setitem__'):
            if entry_id in journal_writer:
                entry = journal_writer[entry_id]
                if hasattr(entry, '__dict__'):
                    for k, v in update_fields.items():
                        setattr(entry, k, v)
                elif isinstance(entry, dict):
                    entry.update(update_fields)
            else:
                journal_writer[entry_id] = update_fields

        # Write stripped-component patterns for failure mining
        if stripped_ids:
            self._write_pattern_data(entry_id, ablation_report)

    def _format_ablation_data(self, report: AblationReport) -> dict:
        """Format ablation report as a dict for journal storage."""
        results = []
        for r in report.ablation_results:
            entry = {
                "component_id": r.component_id if hasattr(r, 'component_id') else r.get('component_id', 0),
                "description": r.component_description if hasattr(r, 'component_description') else r.get('component_description', ''),
                "val_bpb_without": r.val_bpb_without if hasattr(r, 'val_bpb_without') else r.get('val_bpb_without', 0.0),
                "marginal_contribution": r.marginal_contribution if hasattr(r, 'marginal_contribution') else r.get('marginal_contribution', 0.0),
            }
            results.append(entry)

        return {
            "modification_id": report.modification_id,
            "baseline_val_bpb": report.baseline_val_bpb,
            "full_modification_val_bpb": report.full_modification_val_bpb,
            "full_improvement": report.full_improvement,
            "n_components": len(report.components),
            "n_stripped": len(report.stripped_components),
            "ablation_results": results,
            "final_val_bpb": report.final_val_bpb,
        }

    def _format_components(self, report: AblationReport) -> list:
        """Format component list for journal storage."""
        result = []
        for c in report.components:
            entry = {
                "component_id": c.component_id if hasattr(c, 'component_id') else c.get('component_id', 0),
                "description": c.description if hasattr(c, 'description') else c.get('description', ''),
                "category": c.category if hasattr(c, 'category') else c.get('category', ''),
            }
            result.append(entry)
        return result

    def _write_pattern_data(self, entry_id: str, report: AblationReport):
        """Write stripped-component patterns to JSONL for failure mining.

        Each stripped component gets a record indicating it was neutral/negative,
        so the failure mining pipeline can learn to avoid similar changes.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(self.pattern_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Build marginal contribution lookup
        marginal_map = {}
        for r in report.ablation_results:
            cid = r.component_id if hasattr(r, 'component_id') else r.get('component_id', 0)
            mc = r.marginal_contribution if hasattr(r, 'marginal_contribution') else r.get('marginal_contribution', 0.0)
            marginal_map[cid] = mc

        # Build component info lookup
        component_map = {}
        for c in report.components:
            cid = c.component_id if hasattr(c, 'component_id') else c.get('component_id', 0)
            component_map[cid] = {
                "description": c.description if hasattr(c, 'description') else c.get('description', ''),
                "category": c.category if hasattr(c, 'category') else c.get('category', ''),
                "diff": c.diff if hasattr(c, 'diff') else c.get('diff', ''),
            }

        stripped_ids = report.stripped_components
        if isinstance(stripped_ids, list) and stripped_ids and isinstance(stripped_ids[0], dict):
            stripped_ids = [s.get('component_id', 0) for s in stripped_ids]

        records = []
        for cid in stripped_ids:
            comp_info = component_map.get(cid, {})
            records.append({
                "timestamp": time.time(),
                "journal_entry_id": entry_id,
                "modification_id": report.modification_id,
                "component_id": cid,
                "category": comp_info.get("category", "unknown"),
                "description": comp_info.get("description", ""),
                "marginal_contribution": marginal_map.get(cid, 0.0),
                "baseline_val_bpb": report.baseline_val_bpb,
                "full_improvement": report.full_improvement,
                "pattern_type": "stripped_neutral" if marginal_map.get(cid, 0.0) == 0.0 else "stripped_negative",
            })

        with open(self.pattern_output_path, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
