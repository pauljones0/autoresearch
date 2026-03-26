"""
Bandit dashboard: CLI and HTML renderers for bandit state and analysis.
"""

from bandit.schemas import BanditState, ArmState, ABAnalysisReport


class BanditDashboard:
    """Renders bandit state and analysis as CLI text or HTML."""

    # ------------------------------------------------------------------
    # CLI Renderer (80-column terminal)
    # ------------------------------------------------------------------

    def render_cli(
        self,
        state: BanditState,
        analysis: ABAnalysisReport = None,
    ) -> str:
        """Render an 80-column CLI dashboard."""
        lines = []
        w = 80
        lines.append("=" * w)
        lines.append("ADAPTIVE BANDIT DASHBOARD".center(w))
        lines.append("=" * w)

        # Regime and global stats
        lines.append(f"  Regime: {state.regime:<24} "
                     f"Iteration: {state.global_iteration}")
        lines.append(f"  T_base: {state.T_base:<10.4f} "
                     f"Min T: {state.min_temperature:<10.4f} "
                     f"Reheat Factor: {state.reheat_factor}")
        lines.append(f"  Exploration Floor: {state.exploration_floor:<8.3f} "
                     f"Paper Pref: {state.paper_preference_ratio:.2f}")
        lines.append("-" * w)

        # Arm inventory table
        lines.append("  ARM INVENTORY")
        lines.append(f"  {'Arm':<20} {'Alpha':>6} {'Beta':>6} "
                     f"{'Att':>5} {'Suc':>5} {'T':>8} {'Boost':>6} {'Fails':>5}")
        lines.append("  " + "-" * 69)

        for arm_id, arm in sorted(state.arms.items()):
            if not isinstance(arm, ArmState):
                continue
            lines.append(
                f"  {arm_id:<20} {arm.alpha:>6.1f} {arm.beta:>6.1f} "
                f"{arm.total_attempts:>5} {arm.total_successes:>5} "
                f"{arm.temperature:>8.4f} {arm.diagnostics_boost:>6.2f} "
                f"{arm.consecutive_failures:>5}")

        lines.append("-" * w)

        # Selection history summary
        lines.append("  SELECTION HISTORY")
        total_attempts = sum(
            a.total_attempts for a in state.arms.values()
            if isinstance(a, ArmState))
        if total_attempts > 0:
            for arm_id, arm in sorted(state.arms.items()):
                if not isinstance(arm, ArmState) or arm.total_attempts == 0:
                    continue
                pct = arm.total_attempts / total_attempts * 100
                bar_len = int(pct / 100 * 40)
                bar = "#" * bar_len + "." * (40 - bar_len)
                lines.append(f"  {arm_id:<20} [{bar}] {pct:>5.1f}%")
        else:
            lines.append("  No selections yet.")

        lines.append("-" * w)

        # Annealing monitor
        lines.append("  ANNEALING MONITOR")
        for arm_id, arm in sorted(state.arms.items()):
            if not isinstance(arm, ArmState):
                continue
            reheat_info = f"reheats={arm.reheat_count}" if arm.reheat_count > 0 else ""
            lines.append(
                f"  {arm_id:<20} T={arm.temperature:.5f} "
                f"consec_fail={arm.consecutive_failures} {reheat_info}")

        lines.append("-" * w)

        # Temperature trajectory (simplified)
        lines.append("  TEMPERATURE TRAJECTORY")
        temps = []
        for arm_id, arm in sorted(state.arms.items()):
            if isinstance(arm, ArmState):
                temps.append((arm_id, arm.temperature))
        if temps:
            max_t = max(t for _, t in temps) or 1.0
            for arm_id, t in temps:
                bar_len = int(t / max_t * 40) if max_t > 0 else 0
                bar = "|" * bar_len
                lines.append(f"  {arm_id:<20} {bar} {t:.5f}")

        lines.append("-" * w)

        # Allocation efficiency (if analysis available)
        if analysis is not None:
            lines.append("  ALLOCATION EFFICIENCY (A/B)")
            lines.append(f"  Treatment median: {analysis.treatment_median_improvement:.4f}")
            lines.append(f"  Control median:   {analysis.control_median_improvement:.4f}")
            lines.append(f"  U-statistic:      {analysis.u_statistic:.2f}")
            lines.append(f"  p-value:          {analysis.p_value:.4f}")
            lines.append(f"  Effect size:      {analysis.effect_size:.4f}")
            lines.append(f"  Verdict:          {analysis.verdict}")
            lines.append(f"  Waste (treat):    {analysis.waste_rate_treatment:.1%}")
            lines.append(f"  Waste (ctrl):     {analysis.waste_rate_control:.1%}")
            lines.append(f"  Stepping stones:  {analysis.annealing_stepping_stones}")
            lines.append("-" * w)

        # Health alerts
        lines.append("  HEALTH ALERTS")
        issues = state.validate()
        if issues:
            for issue in issues[:10]:
                lines.append(f"  [!] {issue}")
        else:
            lines.append("  All clear.")

        lines.append("=" * w)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML Renderer
    # ------------------------------------------------------------------

    def render_html(
        self,
        state: BanditState,
        analysis: ABAnalysisReport = None,
    ) -> str:
        """Render an HTML dashboard with sections for interactive viewing."""
        parts = []
        parts.append("<!DOCTYPE html>")
        parts.append("<html><head><meta charset='utf-8'>")
        parts.append("<title>Adaptive Bandit Dashboard</title>")
        parts.append("<style>")
        parts.append("body { font-family: monospace; margin: 20px; background: #f5f5f5; }")
        parts.append("h1 { text-align: center; }")
        parts.append("h2 { border-bottom: 2px solid #333; padding-bottom: 4px; }")
        parts.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        parts.append("th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: right; }")
        parts.append("th { background: #ddd; }")
        parts.append("td:first-child, th:first-child { text-align: left; }")
        parts.append(".bar { background: #4a90d9; height: 16px; display: inline-block; }")
        parts.append(".alert { color: #c00; font-weight: bold; }")
        parts.append(".ok { color: #0a0; }")
        parts.append("section { background: #fff; padding: 16px; margin-bottom: 16px; "
                     "border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }")
        parts.append("</style></head><body>")

        parts.append("<h1>Adaptive Bandit Dashboard</h1>")

        # Regime / Global Stats
        parts.append("<section><h2>Global State</h2>")
        parts.append("<table>")
        parts.append(f"<tr><td>Regime</td><td>{_esc(state.regime)}</td></tr>")
        parts.append(f"<tr><td>Iteration</td><td>{state.global_iteration}</td></tr>")
        parts.append(f"<tr><td>T_base</td><td>{state.T_base:.4f}</td></tr>")
        parts.append(f"<tr><td>Min Temperature</td><td>{state.min_temperature:.4f}</td></tr>")
        parts.append(f"<tr><td>Reheat Factor</td><td>{state.reheat_factor}</td></tr>")
        parts.append(f"<tr><td>Exploration Floor</td><td>{state.exploration_floor:.3f}</td></tr>")
        parts.append(f"<tr><td>Paper Preference</td><td>{state.paper_preference_ratio:.2f}</td></tr>")
        parts.append("</table></section>")

        # Arm inventory
        parts.append("<section><h2>Arm Inventory</h2>")
        parts.append("<table><tr><th>Arm</th><th>Alpha</th><th>Beta</th>"
                     "<th>Attempts</th><th>Successes</th><th>Temperature</th>"
                     "<th>Boost</th><th>Consec Fails</th></tr>")
        for arm_id, arm in sorted(state.arms.items()):
            if not isinstance(arm, ArmState):
                continue
            parts.append(
                f"<tr><td>{_esc(arm_id)}</td>"
                f"<td>{arm.alpha:.1f}</td><td>{arm.beta:.1f}</td>"
                f"<td>{arm.total_attempts}</td><td>{arm.total_successes}</td>"
                f"<td>{arm.temperature:.4f}</td>"
                f"<td>{arm.diagnostics_boost:.2f}</td>"
                f"<td>{arm.consecutive_failures}</td></tr>")
        parts.append("</table></section>")

        # Selection history
        parts.append("<section><h2>Selection History</h2>")
        total_attempts = sum(
            a.total_attempts for a in state.arms.values()
            if isinstance(a, ArmState))
        if total_attempts > 0:
            for arm_id, arm in sorted(state.arms.items()):
                if not isinstance(arm, ArmState) or arm.total_attempts == 0:
                    continue
                pct = arm.total_attempts / total_attempts * 100
                bar_w = int(pct / 100 * 300)
                parts.append(
                    f"<div>{_esc(arm_id)}: "
                    f"<span class='bar' style='width:{bar_w}px'></span> "
                    f"{pct:.1f}%</div>")
        else:
            parts.append("<p>No selections yet.</p>")
        parts.append("</section>")

        # Annealing monitor
        parts.append("<section><h2>Annealing Monitor</h2>")
        parts.append("<table><tr><th>Arm</th><th>Temperature</th>"
                     "<th>Consec Fails</th><th>Reheats</th></tr>")
        for arm_id, arm in sorted(state.arms.items()):
            if not isinstance(arm, ArmState):
                continue
            parts.append(
                f"<tr><td>{_esc(arm_id)}</td>"
                f"<td>{arm.temperature:.5f}</td>"
                f"<td>{arm.consecutive_failures}</td>"
                f"<td>{arm.reheat_count}</td></tr>")
        parts.append("</table></section>")

        # Temperature trajectory
        parts.append("<section><h2>Temperature Trajectory</h2>")
        temps = [(aid, a.temperature) for aid, a in sorted(state.arms.items())
                 if isinstance(a, ArmState)]
        if temps:
            max_t = max(t for _, t in temps) or 1.0
            for arm_id, t in temps:
                bar_w = int(t / max_t * 300) if max_t > 0 else 0
                parts.append(
                    f"<div>{_esc(arm_id)}: "
                    f"<span class='bar' style='width:{bar_w}px'></span> "
                    f"{t:.5f}</div>")
        parts.append("</section>")

        # A/B analysis
        if analysis is not None:
            parts.append("<section><h2>A/B Test Results</h2>")
            parts.append("<table>")
            parts.append(f"<tr><td>Treatment Median</td><td>{analysis.treatment_median_improvement:.4f}</td></tr>")
            parts.append(f"<tr><td>Control Median</td><td>{analysis.control_median_improvement:.4f}</td></tr>")
            parts.append(f"<tr><td>U-Statistic</td><td>{analysis.u_statistic:.2f}</td></tr>")
            parts.append(f"<tr><td>p-value</td><td>{analysis.p_value:.4f}</td></tr>")
            parts.append(f"<tr><td>Effect Size</td><td>{analysis.effect_size:.4f}</td></tr>")
            parts.append(f"<tr><td>Verdict</td><td>{_esc(analysis.verdict)}</td></tr>")
            parts.append(f"<tr><td>Waste (Treatment)</td><td>{analysis.waste_rate_treatment:.1%}</td></tr>")
            parts.append(f"<tr><td>Waste (Control)</td><td>{analysis.waste_rate_control:.1%}</td></tr>")
            parts.append(f"<tr><td>Stepping Stones</td><td>{analysis.annealing_stepping_stones}</td></tr>")
            parts.append("</table></section>")

        # Health alerts
        parts.append("<section><h2>Health Alerts</h2>")
        issues = state.validate()
        if issues:
            for issue in issues[:10]:
                parts.append(f"<p class='alert'>{_esc(issue)}</p>")
        else:
            parts.append("<p class='ok'>All clear.</p>")
        parts.append("</section>")

        parts.append("</body></html>")
        return "\n".join(parts)


def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))
