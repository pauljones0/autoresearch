"""
PipelineMonitor — CLI and HTML dashboard for the Model Scientist Pipeline.

Shows: active metrics and correlations, recent critic proposals,
promotion/retirement history, pipeline health.
"""

import json
import os
import time
import argparse
from datetime import datetime


class PipelineMonitor:
    """Monitor and report on pipeline health."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "data"
        )

    def _load_json(self, filename: str) -> dict | list:
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

    def collect_status(self) -> dict:
        """Collect full pipeline status from all data files."""
        status = {
            "timestamp": time.time(),
            "metrics": self._metrics_status(),
            "journal": self._journal_status(),
            "safety": self._safety_status(),
            "failures": self._failure_status(),
        }
        return status

    def _metrics_status(self) -> dict:
        registry = self._load_json("metric_registry.json")
        metrics = registry.get("metrics", []) if isinstance(registry, dict) else registry
        if not isinstance(metrics, list):
            metrics = []

        active = [m for m in metrics if m.get("status") == "active"]
        candidates = [m for m in metrics if m.get("status") == "candidate"]
        retired = [m for m in metrics if m.get("status") == "retired"]

        return {
            "total": len(metrics),
            "active": len(active),
            "candidates": len(candidates),
            "retired": len(retired),
            "active_metrics": [
                {
                    "name": m.get("name", "?"),
                    "correlation": m.get("correlation_with_success", 0),
                    "source": m.get("source", "?"),
                }
                for m in active
            ],
            "candidate_metrics": [
                {"name": m.get("name", "?"), "source": m.get("source", "?")}
                for m in candidates
            ],
        }

    def _journal_status(self) -> dict:
        # Try standard journal path
        from ..journal.schema import JOURNAL_PATH
        entries = []
        if os.path.exists(JOURNAL_PATH):
            try:
                with open(JOURNAL_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
            except IOError:
                pass

        total = len(entries)
        accepted = sum(1 for e in entries if e.get("verdict") == "accepted")
        rejected = sum(1 for e in entries if e.get("verdict") == "rejected")
        crashed = sum(1 for e in entries if e.get("verdict") == "crashed")
        success_rate = accepted / total if total > 0 else 0

        recent = entries[-5:] if entries else []

        return {
            "total_experiments": total,
            "accepted": accepted,
            "rejected": rejected,
            "crashed": crashed,
            "success_rate": f"{success_rate:.1%}",
            "recent": [
                {
                    "id": e.get("id", "?")[:8],
                    "verdict": e.get("verdict", "?"),
                    "delta": e.get("actual_delta", 0),
                    "hypothesis": (e.get("hypothesis", "")[:60] + "...")
                    if len(e.get("hypothesis", "")) > 60
                    else e.get("hypothesis", ""),
                }
                for e in recent
            ],
        }

    def _safety_status(self) -> dict:
        state = self._load_json("safety_state.json")
        if not state:
            return {"loaded": False}
        return {
            "loaded": True,
            "ablation_compute_used_s": state.get("ablation_compute_used", 0),
            "scale_test_compute_used_s": state.get("scale_test_compute_used", 0),
            "total_training_compute_s": state.get("total_training_compute", 0),
            "review_queue_size": len(
                [r for r in state.get("review_queue", []) if r.get("status") == "pending"]
            ),
        }

    def _failure_status(self) -> dict:
        patterns_path = os.path.join(self.data_dir, "failure_patterns.json")
        if not os.path.exists(patterns_path):
            return {"patterns": 0}
        try:
            with open(patterns_path) as f:
                patterns = json.load(f)
            if isinstance(patterns, list):
                return {
                    "patterns": len(patterns),
                    "top_patterns": [
                        {
                            "description": p.get("description", "?")[:80],
                            "instance_count": p.get("instance_count", 0),
                        }
                        for p in sorted(
                            patterns,
                            key=lambda x: x.get("instance_count", 0),
                            reverse=True,
                        )[:5]
                    ],
                }
        except (json.JSONDecodeError, IOError):
            pass
        return {"patterns": 0}

    # --- CLI output ---

    def print_status(self):
        """Print a formatted CLI dashboard."""
        status = self.collect_status()
        ts = datetime.fromtimestamp(status["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*60}")
        print(f"  Model Scientist Pipeline — Status @ {ts}")
        print(f"{'='*60}")

        # Journal
        j = status["journal"]
        print(f"\n--- Experiment Journal ---")
        print(f"  Total: {j['total_experiments']}  |  "
              f"Accepted: {j['accepted']}  |  "
              f"Rejected: {j['rejected']}  |  "
              f"Crashed: {j['crashed']}  |  "
              f"Success rate: {j['success_rate']}")
        if j.get("recent"):
            print(f"  Recent:")
            for r in j["recent"]:
                delta_str = f"{r['delta']:+.6f}" if r["delta"] else "N/A"
                print(f"    [{r['verdict']:8s}] delta={delta_str}  {r['hypothesis']}")

        # Metrics
        m = status["metrics"]
        print(f"\n--- Metric Registry ---")
        print(f"  Active: {m['active']}  |  Candidates: {m['candidates']}  |  Retired: {m['retired']}")
        if m.get("active_metrics"):
            print(f"  Active metrics:")
            for am in m["active_metrics"]:
                print(f"    {am['name']:30s}  r={am['correlation']:.3f}  ({am['source']})")

        # Failures
        fp = status["failures"]
        print(f"\n--- Failure Patterns ---")
        print(f"  Known patterns: {fp.get('patterns', 0)}")
        if fp.get("top_patterns"):
            for p in fp["top_patterns"]:
                print(f"    [{p['instance_count']} instances] {p['description']}")

        # Safety
        s = status["safety"]
        print(f"\n--- Safety Guard ---")
        if s.get("loaded"):
            train_t = s.get("total_training_compute_s", 0)
            print(f"  Training compute baseline: {train_t:.0f}s")
            print(f"  Ablation compute used: {s.get('ablation_compute_used_s', 0):.0f}s")
            print(f"  Scale test compute used: {s.get('scale_test_compute_used_s', 0):.0f}s")
            print(f"  Metrics pending review: {s.get('review_queue_size', 0)}")
        else:
            print(f"  No safety state loaded (first run?)")

        print(f"\n{'='*60}\n")

    # --- HTML output ---

    def generate_html(self, output_path: str = None) -> str:
        """Generate an HTML dashboard page."""
        status = self.collect_status()
        ts = datetime.fromtimestamp(status["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        j = status["journal"]
        m = status["metrics"]
        fp = status["failures"]
        s = status["safety"]

        # Build metric rows
        metric_rows = ""
        for am in m.get("active_metrics", []):
            r_val = am["correlation"]
            color = "#2ecc71" if r_val > 0.3 else "#f39c12" if r_val > 0.1 else "#e74c3c"
            metric_rows += (
                f'<tr><td>{am["name"]}</td>'
                f'<td style="color:{color};font-weight:bold">{r_val:.3f}</td>'
                f'<td>{am["source"]}</td></tr>\n'
            )

        # Build recent experiment rows
        recent_rows = ""
        for r in j.get("recent", []):
            v = r["verdict"]
            vcolor = {"accepted": "#2ecc71", "rejected": "#e74c3c", "crashed": "#9b59b6"}.get(v, "#888")
            delta_str = f'{r["delta"]:+.6f}' if r["delta"] else "N/A"
            recent_rows += (
                f'<tr><td style="color:{vcolor}">{v}</td>'
                f'<td>{delta_str}</td><td>{r["hypothesis"]}</td></tr>\n'
            )

        # Build failure pattern rows
        pattern_rows = ""
        for p in fp.get("top_patterns", []):
            pattern_rows += f'<tr><td>{p["instance_count"]}</td><td>{p["description"]}</td></tr>\n'

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Model Scientist Pipeline Dashboard</title>
<meta http-equiv="refresh" content="30">
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #16c79a; }}
  h2 {{ color: #e2b93d; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .card {{ background: #222244; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  .stat {{ display: inline-block; margin-right: 24px; }}
  .stat .value {{ font-size: 28px; font-weight: bold; color: #16c79a; }}
  .stat .label {{ font-size: 12px; color: #888; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #333; }}
  th {{ color: #e2b93d; font-size: 12px; text-transform: uppercase; }}
  .ts {{ color: #666; font-size: 13px; }}
</style>
</head>
<body>
<h1>Model Scientist Pipeline</h1>
<p class="ts">Last updated: {ts} (auto-refreshes every 30s)</p>

<h2>Experiment Journal</h2>
<div class="card">
  <div class="stat"><div class="value">{j['total_experiments']}</div><div class="label">Total</div></div>
  <div class="stat"><div class="value" style="color:#2ecc71">{j['accepted']}</div><div class="label">Accepted</div></div>
  <div class="stat"><div class="value" style="color:#e74c3c">{j['rejected']}</div><div class="label">Rejected</div></div>
  <div class="stat"><div class="value" style="color:#9b59b6">{j['crashed']}</div><div class="label">Crashed</div></div>
  <div class="stat"><div class="value">{j['success_rate']}</div><div class="label">Success Rate</div></div>
</div>
{f'''<div class="card">
<table><tr><th>Verdict</th><th>Delta</th><th>Hypothesis</th></tr>
{recent_rows}</table>
</div>''' if recent_rows else ''}

<h2>Metric Registry</h2>
<div class="card">
  <div class="stat"><div class="value">{m['active']}</div><div class="label">Active</div></div>
  <div class="stat"><div class="value">{m['candidates']}</div><div class="label">Candidates</div></div>
  <div class="stat"><div class="value">{m['retired']}</div><div class="label">Retired</div></div>
</div>
{f'''<div class="card">
<table><tr><th>Metric</th><th>Correlation (r)</th><th>Source</th></tr>
{metric_rows}</table>
</div>''' if metric_rows else ''}

<h2>Failure Patterns</h2>
<div class="card">
  <div class="stat"><div class="value">{fp.get('patterns', 0)}</div><div class="label">Known Patterns</div></div>
</div>
{f'''<div class="card">
<table><tr><th>Instances</th><th>Description</th></tr>
{pattern_rows}</table>
</div>''' if pattern_rows else ''}

<h2>Safety Guard</h2>
<div class="card">
  <div class="stat"><div class="value">{s.get('total_training_compute_s', 0):.0f}s</div><div class="label">Training Compute</div></div>
  <div class="stat"><div class="value">{s.get('ablation_compute_used_s', 0):.0f}s</div><div class="label">Ablation Used</div></div>
  <div class="stat"><div class="value">{s.get('scale_test_compute_used_s', 0):.0f}s</div><div class="label">Scale Test Used</div></div>
  <div class="stat"><div class="value">{s.get('review_queue_size', 0)}</div><div class="label">Pending Reviews</div></div>
</div>

</body>
</html>"""

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                f.write(html)

        return html


def main():
    parser = argparse.ArgumentParser(description="Model Scientist Pipeline Monitor")
    parser.add_argument("--data-dir", default=None, help="Data directory path")
    parser.add_argument("--html", default=None, help="Output HTML dashboard to file")
    args = parser.parse_args()

    monitor = PipelineMonitor(data_dir=args.data_dir)
    monitor.print_status()

    if args.html:
        monitor.generate_html(args.html)
        print(f"HTML dashboard written to: {args.html}")


if __name__ == "__main__":
    main()
