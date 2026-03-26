"""
Meta-optimization dashboard for CLI and HTML rendering.

Provides visibility into the meta-loop's regime, budget, posteriors,
experiment history, active configuration, STOP strategies, convergence
status, and knowledge base summary.
"""

import time
import html as html_lib

from meta.schemas import (
    MetaBanditState,
    DimensionState,
    ROIData,
    MetaExperimentResult,
)


class MetaDashboard:
    """Render meta-optimization status as CLI or HTML."""

    CLI_WIDTH = 80

    def render_cli(self, meta_state: MetaBanditState,
                   roi_data: ROIData = None,
                   experiment_history: list = None,
                   stop_strategies: list = None,
                   convergence_status=None,
                   knowledge_summary: dict = None,
                   meta_parameters: list = None) -> str:
        """Render a CLI dashboard string (80-col terminal).

        Args:
            meta_state: Current meta-bandit state.
            roi_data: Optional ROI tracking data.
            experiment_history: Optional list of MetaExperimentResult dicts.
            stop_strategies: Optional list of active STOP strategy dicts.
            convergence_status: Optional ConvergenceStatus object/dict.
            knowledge_summary: Optional dict with knowledge base stats.
            meta_parameters: Optional list of MetaParameter for defaults.

        Returns:
            Formatted string for terminal display.
        """
        lines = []
        w = self.CLI_WIDTH
        lines.append("=" * w)
        lines.append(self._center("META-AUTORESEARCH DASHBOARD", w))
        lines.append("=" * w)

        # Section 1: Meta-Regime & Budget
        lines.append("")
        lines.append(self._section_header("1. Meta-Regime & Budget", w))
        lines.append(f"  Regime:           {meta_state.meta_regime}")
        lines.append(f"  Budget fraction:  {meta_state.budget_fraction:.1%}")
        max_budget = meta_state.budget_fraction * meta_state.budget_cycle_length
        utilization = (
            meta_state.budget_used / max_budget * 100 if max_budget > 0 else 0
        )
        lines.append(f"  Budget used:      {meta_state.budget_used:.0f}"
                      f" / {max_budget:.0f}"
                      f" ({utilization:.1f}% utilized)")
        lines.append(f"  Total experiments:{meta_state.total_meta_experiments}")
        if roi_data:
            roi = roi_data if isinstance(roi_data, dict) else roi_data.to_dict()
            lines.append(f"  ROI:              {roi.get('roi', 0):.2f}")
            lines.append(f"  val_bpb gain:     "
                          f"{roi.get('cumulative_val_bpb_improvement', 0):.4f}")

        # Section 2: Dimension Posteriors
        lines.append("")
        lines.append(self._section_header("2. Dimension Posteriors", w))
        hdr = f"  {'Param':<22} {'Best':>8} {'Mean':>8} {'95% CI':>16}"
        hdr += f" {'Prom':>4} {'Sens':>6}"
        lines.append(hdr)
        lines.append("  " + "-" * (w - 4))

        for pid, dim in sorted(meta_state.dimensions.items()):
            if not isinstance(dim, DimensionState):
                continue
            best_str = str(dim.current_best)[:8]
            mean, ci_lo, ci_hi = self._posterior_summary(dim)
            promoted = "Y" if dim.last_promoted > 0 else "-"
            sens = self._sensitivity_label(pid, meta_parameters)
            ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]"
            line = (f"  {pid:<22} {best_str:>8} {mean:>8.3f} "
                    f"{ci_str:>16} {promoted:>4} {sens:>6}")
            lines.append(line)

        # Section 3: Meta-Experiment History (last 20)
        lines.append("")
        lines.append(self._section_header("3. Meta-Experiment History", w))
        history = experiment_history or []
        recent = history[-20:] if len(history) > 20 else history
        if not recent:
            lines.append("  (no experiments recorded)")
        else:
            hdr = f"  {'ID':<12} {'IR':>8} {'vs Base':>10} {'Dims':>6}"
            lines.append(hdr)
            lines.append("  " + "-" * (w - 4))
            for exp in recent:
                e = exp if isinstance(exp, dict) else exp.to_dict()
                eid = str(e.get("experiment_id", ""))[:12]
                ir = e.get("improvement_rate", 0)
                comp = e.get("compared_to_baseline", "?")
                ndiffs = len(e.get("config_diff", []))
                lines.append(f"  {eid:<12} {ir:>8.4f} {comp:>10} {ndiffs:>6}")

        # Section 4: Active Configuration (diff from defaults)
        lines.append("")
        lines.append(self._section_header("4. Active Configuration", w))
        defaults = {}
        if meta_parameters:
            for p in meta_parameters:
                defaults[p.param_id] = p.default_value

        diffs_found = False
        for pid, val in sorted(meta_state.best_config.items()):
            default_val = defaults.get(pid)
            if val != default_val:
                lines.append(f"  {pid}: {default_val} -> {val}")
                diffs_found = True
        if not diffs_found:
            lines.append("  (no changes from defaults)")

        # Section 5: STOP Strategies
        lines.append("")
        lines.append(self._section_header("5. STOP Strategies", w))
        strategies = stop_strategies or []
        if not strategies:
            lines.append("  (no active strategies)")
        else:
            for s in strategies:
                sd = s if isinstance(s, dict) else s.to_dict()
                sid = sd.get("strategy_id", "?")
                hook = sd.get("hook_type", "?")
                desc = sd.get("description", "")[:40]
                lines.append(f"  [{hook}] {sid}: {desc}")

        # Section 6: Convergence Status
        lines.append("")
        lines.append(self._section_header("6. Convergence Status", w))
        if convergence_status:
            cs = (convergence_status if isinstance(convergence_status, dict)
                  else convergence_status.to_dict())
            lines.append(f"  Converged:          "
                          f"{cs.get('converged', False)}")
            lines.append(f"  Exps since promo:   "
                          f"{cs.get('meta_experiments_since_last_promotion', 0)}")
            lines.append(f"  Max posterior var:   "
                          f"{cs.get('max_posterior_variance', 0):.4f}")
            lines.append(f"  Recommendation:     "
                          f"{cs.get('recommendation', '?')}")
        else:
            lines.append("  (no convergence data)")

        # Section 7: Knowledge Base Summary
        lines.append("")
        lines.append(self._section_header("7. Knowledge Base Summary", w))
        if knowledge_summary:
            lines.append(f"  Total insights:     "
                          f"{knowledge_summary.get('total_insights', 0)}")
            lines.append(f"  High confidence:    "
                          f"{knowledge_summary.get('high_confidence', 0)}")
            lines.append(f"  Universal:          "
                          f"{knowledge_summary.get('universal', 0)}")
            lines.append(f"  Transferable:       "
                          f"{knowledge_summary.get('transferable', 0)}")
        else:
            lines.append("  (no knowledge base)")

        lines.append("")
        lines.append("=" * w)
        return "\n".join(lines)

    def render_html(self, meta_state: MetaBanditState,
                    roi_data: ROIData = None,
                    experiment_history: list = None,
                    stop_strategies: list = None,
                    convergence_status=None,
                    knowledge_summary: dict = None,
                    meta_parameters: list = None) -> str:
        """Render an HTML dashboard.

        Same data as render_cli, but formatted as styled HTML sections.

        Returns:
            HTML string.
        """
        sections = []

        # CSS
        css = (
            "<style>"
            "body{font-family:monospace;max-width:900px;margin:auto;padding:20px}"
            "h1{text-align:center;border-bottom:2px solid #333}"
            "h2{background:#eee;padding:6px 10px;margin-top:20px}"
            "table{border-collapse:collapse;width:100%;margin:8px 0}"
            "th,td{border:1px solid #ccc;padding:4px 8px;text-align:right}"
            "th{background:#ddd}"
            "td:first-child,th:first-child{text-align:left}"
            ".badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "font-weight:bold}"
            ".regime-baseline{background:#ffc;color:#660}"
            ".regime-active{background:#cfc;color:#060}"
            ".regime-maintenance{background:#ccf;color:#006}"
            ".diff-changed{color:#c00}"
            "</style>"
        )

        sections.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        sections.append("<title>Meta-Autoresearch Dashboard</title>")
        sections.append(css)
        sections.append("</head><body>")
        sections.append("<h1>Meta-Autoresearch Dashboard</h1>")

        # Section 1: Regime & Budget
        regime = html_lib.escape(meta_state.meta_regime)
        regime_cls = f"regime-{regime}"
        sections.append("<h2>1. Meta-Regime &amp; Budget</h2>")
        sections.append(f"<p>Regime: <span class='badge {regime_cls}'>"
                        f"{regime}</span></p>")
        max_budget = meta_state.budget_fraction * meta_state.budget_cycle_length
        util = meta_state.budget_used / max_budget * 100 if max_budget > 0 else 0
        sections.append("<table><tr><th>Metric</th><th>Value</th></tr>")
        sections.append(f"<tr><td>Budget fraction</td>"
                        f"<td>{meta_state.budget_fraction:.1%}</td></tr>")
        sections.append(f"<tr><td>Budget used</td>"
                        f"<td>{meta_state.budget_used:.0f} / "
                        f"{max_budget:.0f} ({util:.1f}%)</td></tr>")
        sections.append(f"<tr><td>Total experiments</td>"
                        f"<td>{meta_state.total_meta_experiments}</td></tr>")
        if roi_data:
            roi = roi_data if isinstance(roi_data, dict) else roi_data.to_dict()
            sections.append(f"<tr><td>ROI</td>"
                            f"<td>{roi.get('roi', 0):.2f}</td></tr>")
            sections.append(f"<tr><td>val_bpb gain</td><td>"
                            f"{roi.get('cumulative_val_bpb_improvement', 0):.4f}"
                            f"</td></tr>")
        sections.append("</table>")

        # Section 2: Dimension Posteriors
        sections.append("<h2>2. Dimension Posteriors</h2>")
        sections.append("<table><tr><th>Param</th><th>Best</th><th>Mean</th>"
                        "<th>95% CI</th><th>Promoted</th><th>Sens</th></tr>")
        for pid, dim in sorted(meta_state.dimensions.items()):
            if not isinstance(dim, DimensionState):
                continue
            best_str = html_lib.escape(str(dim.current_best)[:12])
            mean, ci_lo, ci_hi = self._posterior_summary(dim)
            promoted = "Y" if dim.last_promoted > 0 else "-"
            sens = self._sensitivity_label(pid, meta_parameters)
            sections.append(
                f"<tr><td>{html_lib.escape(pid)}</td>"
                f"<td>{best_str}</td>"
                f"<td>{mean:.3f}</td>"
                f"<td>[{ci_lo:.3f}, {ci_hi:.3f}]</td>"
                f"<td>{promoted}</td>"
                f"<td>{html_lib.escape(sens)}</td></tr>"
            )
        sections.append("</table>")

        # Section 3: Experiment History
        sections.append("<h2>3. Meta-Experiment History (last 20)</h2>")
        history = experiment_history or []
        recent = history[-20:] if len(history) > 20 else history
        if not recent:
            sections.append("<p><em>No experiments recorded.</em></p>")
        else:
            sections.append("<table><tr><th>ID</th><th>IR</th>"
                            "<th>vs Baseline</th><th>Dims</th></tr>")
            for exp in recent:
                e = exp if isinstance(exp, dict) else exp.to_dict()
                eid = html_lib.escape(str(e.get("experiment_id", ""))[:20])
                ir = e.get("improvement_rate", 0)
                comp = html_lib.escape(str(e.get("compared_to_baseline", "?")))
                ndiffs = len(e.get("config_diff", []))
                sections.append(f"<tr><td>{eid}</td><td>{ir:.4f}</td>"
                                f"<td>{comp}</td><td>{ndiffs}</td></tr>")
            sections.append("</table>")

        # Section 4: Active Configuration
        sections.append("<h2>4. Active Configuration</h2>")
        defaults = {}
        if meta_parameters:
            for p in meta_parameters:
                defaults[p.param_id] = p.default_value
        diffs_found = False
        sections.append("<table><tr><th>Param</th><th>Default</th>"
                        "<th>Current</th></tr>")
        for pid, val in sorted(meta_state.best_config.items()):
            default_val = defaults.get(pid)
            if val != default_val:
                sections.append(
                    f"<tr class='diff-changed'>"
                    f"<td>{html_lib.escape(pid)}</td>"
                    f"<td>{html_lib.escape(str(default_val))}</td>"
                    f"<td>{html_lib.escape(str(val))}</td></tr>"
                )
                diffs_found = True
        if not diffs_found:
            sections.append("<tr><td colspan='3'><em>No changes from "
                            "defaults</em></td></tr>")
        sections.append("</table>")

        # Section 5: STOP Strategies
        sections.append("<h2>5. STOP Strategies</h2>")
        strategies = stop_strategies or []
        if not strategies:
            sections.append("<p><em>No active strategies.</em></p>")
        else:
            sections.append("<table><tr><th>Hook</th><th>ID</th>"
                            "<th>Description</th></tr>")
            for s in strategies:
                sd = s if isinstance(s, dict) else s.to_dict()
                sections.append(
                    f"<tr><td>{html_lib.escape(str(sd.get('hook_type', '')))}"
                    f"</td><td>{html_lib.escape(str(sd.get('strategy_id', '')))}"
                    f"</td><td>{html_lib.escape(str(sd.get('description', '')))}"
                    f"</td></tr>"
                )
            sections.append("</table>")

        # Section 6: Convergence
        sections.append("<h2>6. Convergence Status</h2>")
        if convergence_status:
            cs = (convergence_status if isinstance(convergence_status, dict)
                  else convergence_status.to_dict())
            sections.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            sections.append(f"<tr><td>Converged</td>"
                            f"<td>{cs.get('converged', False)}</td></tr>")
            sections.append(f"<tr><td>Experiments since promotion</td>"
                            f"<td>{cs.get('meta_experiments_since_last_promotion', 0)}"
                            f"</td></tr>")
            sections.append(f"<tr><td>Max posterior variance</td>"
                            f"<td>{cs.get('max_posterior_variance', 0):.4f}"
                            f"</td></tr>")
            sections.append(f"<tr><td>Recommendation</td>"
                            f"<td>{html_lib.escape(str(cs.get('recommendation', '?')))}"
                            f"</td></tr>")
            sections.append("</table>")
        else:
            sections.append("<p><em>No convergence data.</em></p>")

        # Section 7: Knowledge Base
        sections.append("<h2>7. Knowledge Base Summary</h2>")
        if knowledge_summary:
            sections.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            for k, v in knowledge_summary.items():
                sections.append(f"<tr><td>{html_lib.escape(str(k))}</td>"
                                f"<td>{v}</td></tr>")
            sections.append("</table>")
        else:
            sections.append("<p><em>No knowledge base.</em></p>")

        sections.append("</body></html>")
        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _center(text: str, width: int) -> str:
        pad = max(0, (width - len(text)) // 2)
        return " " * pad + text

    @staticmethod
    def _section_header(title: str, width: int) -> str:
        return f"--- {title} " + "-" * max(0, width - len(title) - 5)

    @staticmethod
    def _posterior_summary(dim: DimensionState) -> tuple:
        """Compute mean and 95% CI from Beta posteriors of current_best variant.

        Returns (mean, ci_lower, ci_upper).
        """
        var_key = str(dim.current_best)
        post = dim.variant_posteriors.get(var_key, {})
        alpha = post.get("alpha", 1.0) if isinstance(post, dict) else 1.0
        beta = post.get("beta", 1.0) if isinstance(post, dict) else 1.0
        total = alpha + beta
        mean = alpha / total if total > 0 else 0.5

        # Normal approx CI for Beta
        if total > 2:
            import math
            var = (alpha * beta) / (total * total * (total + 1))
            std = math.sqrt(var)
            ci_lo = max(0.0, mean - 1.96 * std)
            ci_hi = min(1.0, mean + 1.96 * std)
        else:
            ci_lo = 0.0
            ci_hi = 1.0

        return mean, ci_lo, ci_hi

    @staticmethod
    def _sensitivity_label(param_id: str, meta_parameters: list = None) -> str:
        """Get the sensitivity label for a parameter."""
        if not meta_parameters:
            return "?"
        for p in meta_parameters:
            if p.param_id == param_id:
                return p.sensitivity_estimate[:6] if p.sensitivity_estimate else "?"
        return "?"
