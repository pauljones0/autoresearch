"""
Kernel dashboard — CLI and HTML views of GPU kernel pipeline status.

Sections: Active Kernel Inventory, Combined Impact, Runtime Health,
Evolutionary Status, Discovery Queue, Failure Patterns.
"""

import json
import os
import time
from datetime import datetime


class KernelDashboard:
    """CLI and HTML dashboard for the GPU kernel pipeline."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _load_json(self, filename: str):
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _load_jsonl(self, filename: str) -> list:
        path = os.path.join(self.data_dir, filename)
        entries = []
        if not os.path.exists(path):
            return entries
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except IOError:
            pass
        return entries

    def _collect_data(self) -> dict:
        """Collect all dashboard data from files."""
        config = self._load_json("kernel_config.json")
        alerts = self._load_jsonl("runtime_alerts.jsonl")
        disable_log = self._load_jsonl("kernel_disable_log.jsonl")
        recovery_log = self._load_jsonl("recovery_log.jsonl")
        mutation_queue = self._load_jsonl("mutation_queue.jsonl")
        discovery_queue = self._load_jsonl("discovery_queue.jsonl")

        # Load verification reports
        reports = {}
        reports_dir = os.path.join(self.data_dir, "verification_reports")
        if os.path.isdir(reports_dir):
            for fname in os.listdir(reports_dir):
                if fname.endswith(".json"):
                    kid = fname.rsplit(".", 1)[0]
                    rpath = os.path.join(reports_dir, fname)
                    try:
                        with open(rpath) as f:
                            reports[kid] = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        pass

        # Load evolution journal
        evo_journal = self._load_jsonl("evolution_journal.jsonl")

        # Load failure patterns
        failure_patterns = self._load_json("failure_patterns.json")
        if isinstance(failure_patterns, dict) and "patterns" in failure_patterns:
            failure_patterns = failure_patterns["patterns"]
        if not isinstance(failure_patterns, list):
            failure_patterns = []

        return {
            "config": config,
            "alerts": alerts,
            "disable_log": disable_log,
            "recovery_log": recovery_log,
            "mutation_queue": mutation_queue,
            "discovery_queue": discovery_queue,
            "reports": reports,
            "evo_journal": evo_journal,
            "failure_patterns": failure_patterns,
        }

    def _kernel_inventory(self, data: dict) -> list:
        """Build active kernel inventory table."""
        config = data["config"]
        reports = data["reports"]
        inventory = []
        for kid, entry in config.items():
            if not isinstance(entry, dict):
                continue
            row = {
                "kernel_id": kid,
                "enabled": entry.get("enabled", False),
                "backend": entry.get("backend", "?"),
                "speedup": entry.get("speedup", 0.0),
                "memory_savings": entry.get("memory_savings_ratio", 0.0),
                "group_id": entry.get("group_id", ""),
                "last_verified": "",
            }
            report = reports.get(kid, {})
            if report:
                ts = report.get("timestamp", 0)
                if ts:
                    row["last_verified"] = datetime.fromtimestamp(ts).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                row["verdict"] = report.get("verdict", "?")
            else:
                row["verdict"] = "N/A"
            inventory.append(row)
        return inventory

    def _combined_impact(self, data: dict) -> dict:
        """Calculate combined performance impact."""
        config = data["config"]
        total_speedup = 0.0
        active_count = 0
        for entry in config.values():
            if isinstance(entry, dict) and entry.get("enabled", False):
                total_speedup += entry.get("speedup", 0.0)
                active_count += 1
        return {
            "active_kernels": active_count,
            "total_tok_sec_improvement": total_speedup,
        }

    def _runtime_health(self, data: dict) -> dict:
        """Collect runtime health: last 20 checks per kernel."""
        alerts = data["alerts"]
        by_kernel = {}
        for a in alerts:
            kid = a.get("kernel_id", "?")
            by_kernel.setdefault(kid, []).append(a)

        health = {}
        for kid, kernel_alerts in by_kernel.items():
            recent = kernel_alerts[-20:]
            warnings = sum(1 for a in recent if a.get("severity") == "warning")
            criticals = sum(1 for a in recent if a.get("severity") == "critical")
            health[kid] = {
                "recent_checks": len(recent),
                "warnings": warnings,
                "criticals": criticals,
            }
        return health

    # ------------------------------------------------------------------
    # CLI rendering
    # ------------------------------------------------------------------

    def render_cli(self) -> str:
        """Render CLI dashboard."""
        data = self._collect_data()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        sep = "=" * 64

        lines.append(sep)
        lines.append(f"  GPU Kernel Dashboard  --  {ts}")
        lines.append(sep)

        # Active Kernel Inventory
        inventory = self._kernel_inventory(data)
        lines.append("")
        lines.append("--- Active Kernel Inventory ---")
        if inventory:
            lines.append(
                f"  {'ID':<24s} {'Status':<8s} {'Speedup':>8s} {'Verdict':<8s} {'Last Verified'}"
            )
            for row in inventory:
                status = "ON" if row["enabled"] else "OFF"
                lines.append(
                    f"  {row['kernel_id']:<24s} {status:<8s} "
                    f"{row['speedup']:>7.2f}x {row['verdict']:<8s} "
                    f"{row['last_verified']}"
                )
        else:
            lines.append("  (no kernels configured)")

        # Combined Impact
        impact = self._combined_impact(data)
        lines.append("")
        lines.append("--- Combined Impact ---")
        lines.append(f"  Active kernels: {impact['active_kernels']}")
        lines.append(
            f"  Total tok/sec improvement: {impact['total_tok_sec_improvement']:.2f}"
        )

        # Runtime Health
        health = self._runtime_health(data)
        lines.append("")
        lines.append("--- Runtime Health (last 20 checks) ---")
        if health:
            for kid, h in health.items():
                lines.append(
                    f"  {kid:<24s}  checks={h['recent_checks']}  "
                    f"warn={h['warnings']}  crit={h['criticals']}"
                )
        else:
            lines.append("  (no runtime data)")

        # Evolutionary Status
        evo = data["evo_journal"]
        lines.append("")
        lines.append("--- Evolutionary Status ---")
        if evo:
            recent_evo = evo[-5:]
            for e in recent_evo:
                lines.append(
                    f"  gen={e.get('generation', '?')}  "
                    f"parent={e.get('parent_id', '?')[:16]}  "
                    f"best_speedup={e.get('best_speedup', 0):.2f}x  "
                    f"improvement={e.get('improvement_over_parent', 0):.1%}"
                )
        else:
            lines.append("  (no evolution data)")

        # Discovery Queue
        dq = data["discovery_queue"]
        lines.append("")
        lines.append("--- Discovery Queue ---")
        if dq:
            for item in dq[-5:]:
                lines.append(
                    f"  group={item.get('group_id', '?')}  "
                    f"score={item.get('adjusted_score', 0):.2f}  "
                    f"type={item.get('fusion_type', '?')}"
                )
        else:
            lines.append("  (empty)")

        # Failure Patterns
        fp = data["failure_patterns"]
        lines.append("")
        lines.append("--- Failure Patterns ---")
        if fp:
            for p in fp[:5]:
                desc = p.get("description", "?")[:60]
                count = p.get("instance_count", 0)
                lines.append(f"  [{count} instances] {desc}")
        else:
            lines.append("  (none recorded)")

        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def render_html(self) -> str:
        """Render HTML dashboard."""
        data = self._collect_data()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        inventory = self._kernel_inventory(data)
        impact = self._combined_impact(data)
        health = self._runtime_health(data)
        evo = data["evo_journal"]
        dq = data["discovery_queue"]
        fp = data["failure_patterns"]

        # Build table rows
        inv_rows = ""
        for row in inventory:
            status_color = "#2ecc71" if row["enabled"] else "#e74c3c"
            inv_rows += (
                f'<tr>'
                f'<td>{row["kernel_id"]}</td>'
                f'<td style="color:{status_color}">{"ON" if row["enabled"] else "OFF"}</td>'
                f'<td>{row["speedup"]:.2f}x</td>'
                f'<td>{row["memory_savings"]:.1%}</td>'
                f'<td>{row["verdict"]}</td>'
                f'<td>{row["last_verified"]}</td>'
                f'</tr>\n'
            )

        health_rows = ""
        for kid, h in health.items():
            crit_color = "#e74c3c" if h["criticals"] > 0 else "#2ecc71"
            health_rows += (
                f'<tr><td>{kid}</td>'
                f'<td>{h["recent_checks"]}</td>'
                f'<td>{h["warnings"]}</td>'
                f'<td style="color:{crit_color}">{h["criticals"]}</td></tr>\n'
            )

        evo_rows = ""
        for e in evo[-5:]:
            evo_rows += (
                f'<tr><td>{e.get("generation", "?")}</td>'
                f'<td>{e.get("parent_id", "?")[:16]}</td>'
                f'<td>{e.get("best_speedup", 0):.2f}x</td>'
                f'<td>{e.get("improvement_over_parent", 0):.1%}</td></tr>\n'
            )

        dq_rows = ""
        for item in dq[-5:]:
            dq_rows += (
                f'<tr><td>{item.get("group_id", "?")}</td>'
                f'<td>{item.get("adjusted_score", 0):.2f}</td>'
                f'<td>{item.get("fusion_type", "?")}</td></tr>\n'
            )

        fp_rows = ""
        for p in fp[:5]:
            fp_rows += (
                f'<tr><td>{p.get("instance_count", 0)}</td>'
                f'<td>{p.get("description", "?")[:80]}</td></tr>\n'
            )

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GPU Kernel Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; }}
  h2 {{ color: #f0c674; border-bottom: 1px solid #30363d; padding-bottom: 6px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  .stat {{ display: inline-block; margin-right: 24px; text-align: center; }}
  .stat .value {{ font-size: 28px; font-weight: bold; color: #58a6ff; }}
  .stat .label {{ font-size: 12px; color: #8b949e; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #30363d; }}
  th {{ color: #f0c674; font-size: 12px; text-transform: uppercase; }}
  .ts {{ color: #8b949e; font-size: 13px; }}
</style>
</head>
<body>
<h1>GPU Kernel Dashboard</h1>
<p class="ts">Last updated: {ts} (auto-refreshes every 30s)</p>

<h2>Active Kernel Inventory</h2>
<div class="card">
<table>
<tr><th>Kernel ID</th><th>Status</th><th>Speedup</th><th>Mem Savings</th><th>Verdict</th><th>Last Verified</th></tr>
{inv_rows if inv_rows else '<tr><td colspan="6">No kernels configured</td></tr>'}
</table>
</div>

<h2>Combined Impact</h2>
<div class="card">
  <div class="stat"><div class="value">{impact['active_kernels']}</div><div class="label">Active Kernels</div></div>
  <div class="stat"><div class="value">{impact['total_tok_sec_improvement']:.2f}</div><div class="label">Total tok/sec Improvement</div></div>
</div>

<h2>Runtime Health (last 20 checks)</h2>
<div class="card">
<table>
<tr><th>Kernel</th><th>Checks</th><th>Warnings</th><th>Criticals</th></tr>
{health_rows if health_rows else '<tr><td colspan="4">No runtime data</td></tr>'}
</table>
</div>

<h2>Evolutionary Status</h2>
<div class="card">
<table>
<tr><th>Generation</th><th>Parent</th><th>Best Speedup</th><th>Improvement</th></tr>
{evo_rows if evo_rows else '<tr><td colspan="4">No evolution data</td></tr>'}
</table>
</div>

<h2>Discovery Queue</h2>
<div class="card">
<table>
<tr><th>Group ID</th><th>Score</th><th>Fusion Type</th></tr>
{dq_rows if dq_rows else '<tr><td colspan="3">Empty</td></tr>'}
</table>
</div>

<h2>Failure Patterns</h2>
<div class="card">
<table>
<tr><th>Instances</th><th>Description</th></tr>
{fp_rows if fp_rows else '<tr><td colspan="2">None recorded</td></tr>'}
</table>
</div>

</body>
</html>"""

        if output_path:
            try:
                with open(output_path, "w") as f:
                    f.write(html)
            except Exception:
                pass

        return html
