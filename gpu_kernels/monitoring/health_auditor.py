"""
Cross-system health auditor for the GPU kernel pipeline.

Verifies data integrity across config, verification reports, journal,
discovery queue, and kernel file paths.
"""

import json
import os


class CrossSystemHealthAuditor:
    """Audit cross-system data integrity for the kernel pipeline."""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "..", "data"
        )

    def audit(self) -> dict:
        """
        Run all integrity checks.

        Checks:
        1. Every active kernel has a verification report.
        2. Every kernel journal entry has correct tags.
        3. kernel_config.json is valid JSON.
        4. Discovery queue is valid JSON/JSONL.
        5. All referenced kernel paths exist.

        Returns:
            dict with keys:
                all_clear: bool — no issues found
                issues: list of issue description strings
        """
        issues = []

        issues.extend(self._check_config_valid())
        issues.extend(self._check_active_kernels_have_reports())
        issues.extend(self._check_journal_tags())
        issues.extend(self._check_discovery_queue_valid())
        issues.extend(self._check_kernel_paths_exist())

        return {
            "all_clear": len(issues) == 0,
            "issues": issues,
        }

    def _check_config_valid(self) -> list:
        """Check that kernel_config.json is valid JSON."""
        issues = []
        config_path = os.path.join(self.data_dir, "kernel_config.json")
        if not os.path.exists(config_path):
            # Not an issue — may not exist yet
            return issues
        try:
            with open(config_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                issues.append(
                    f"kernel_config.json: expected dict, got {type(data).__name__}"
                )
        except json.JSONDecodeError as e:
            issues.append(f"kernel_config.json: invalid JSON — {e}")
        except IOError as e:
            issues.append(f"kernel_config.json: read error — {e}")
        return issues

    def _check_active_kernels_have_reports(self) -> list:
        """Check that every active kernel has a verification report."""
        issues = []
        config = self._load_json("kernel_config.json")
        if not config:
            return issues

        reports_dir = os.path.join(self.data_dir, "verification_reports")

        for kid, entry in config.items():
            if not isinstance(entry, dict):
                continue
            if not entry.get("enabled", False):
                continue

            # Check for verification report
            report_path = entry.get("verification_report", "")
            if report_path and os.path.isabs(report_path):
                if not os.path.exists(report_path):
                    issues.append(
                        f"Kernel '{kid}': verification report not found at {report_path}"
                    )
            else:
                # Check in standard reports directory
                std_path = os.path.join(reports_dir, f"{kid}.json")
                if not os.path.exists(std_path):
                    issues.append(
                        f"Kernel '{kid}': no verification report found "
                        f"(checked {std_path})"
                    )

        return issues

    def _check_journal_tags(self) -> list:
        """Check that kernel journal entries have correct tags."""
        issues = []
        journal = self._load_jsonl("kernel_journal.jsonl")
        if not journal:
            return issues

        required_fields = ["kernel_id", "timestamp"]
        for i, entry in enumerate(journal):
            for field in required_fields:
                if field not in entry:
                    issues.append(
                        f"kernel_journal.jsonl entry {i}: missing '{field}'"
                    )

            # Check tags are a list if present
            tags = entry.get("tags")
            if tags is not None and not isinstance(tags, list):
                issues.append(
                    f"kernel_journal.jsonl entry {i}: 'tags' should be list, "
                    f"got {type(tags).__name__}"
                )

        return issues

    def _check_discovery_queue_valid(self) -> list:
        """Check that discovery queue files are valid."""
        issues = []

        # Check JSONL queue
        queue_path = os.path.join(self.data_dir, "discovery_queue.jsonl")
        if os.path.exists(queue_path):
            try:
                with open(queue_path) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                            except json.JSONDecodeError:
                                issues.append(
                                    f"discovery_queue.jsonl line {line_num}: invalid JSON"
                                )
            except IOError as e:
                issues.append(f"discovery_queue.jsonl: read error — {e}")

        # Check JSON queue
        queue_json_path = os.path.join(self.data_dir, "discovery_queue.json")
        if os.path.exists(queue_json_path):
            try:
                with open(queue_json_path) as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                issues.append(f"discovery_queue.json: invalid JSON — {e}")
            except IOError as e:
                issues.append(f"discovery_queue.json: read error — {e}")

        return issues

    def _check_kernel_paths_exist(self) -> list:
        """Check that all kernel paths referenced in config exist."""
        issues = []
        config = self._load_json("kernel_config.json")
        if not config:
            return issues

        for kid, entry in config.items():
            if not isinstance(entry, dict):
                continue
            kernel_path = entry.get("kernel_path", "")
            if kernel_path and not os.path.exists(kernel_path):
                issues.append(
                    f"Kernel '{kid}': kernel_path not found — {kernel_path}"
                )

        return issues

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
