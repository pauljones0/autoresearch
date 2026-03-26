"""
Runs a modification at multiple scales sequentially (smallest first).
Generates temporary modified train.py copies with adjusted hyperparameters,
runs them via subprocess, and collects results.
"""

import os
import re
import math
import tempfile
import subprocess
import time
from dataclasses import asdict

from model_scientist.schemas import ScaleConfig, ScalingResult
from model_scientist.scaling.config_deriver import ScaleConfigDeriver


# Hyperparameter lines in train.py that we need to patch
_HP_PATTERNS = {
    "DEPTH": re.compile(r"^(DEPTH\s*=\s*)\d+", re.MULTILINE),
    "DEVICE_BATCH_SIZE": re.compile(r"^(DEVICE_BATCH_SIZE\s*=\s*)\d+", re.MULTILINE),
    "TIME_BUDGET": re.compile(r"^(TIME_BUDGET\s*=\s*)\d+", re.MULTILINE),
}

_VAL_BPB_RE = re.compile(r"val_bpb:\s+([\d.]+)")

MAX_RUN_TIMEOUT = 600  # 10 minutes


class ScaleRunner:
    """Runs training at multiple scales and collects results."""

    def __init__(self, base_train_path: str | None = None, head_dim: int = 128):
        self.head_dim = head_dim
        self.base_train_path = base_train_path
        self._deriver = ScaleConfigDeriver()

    def run_at_scales(
        self,
        modification_diff: str,
        modified_source: str,
        scales: list[float] | None = None,
    ) -> list[ScalingResult]:
        if scales is None:
            scales = [0.25, 0.5, 1.0]

        base_source = self._read_file(self.base_train_path)
        base_depth, base_aspect_ratio, base_head_dim = self._extract_hp(base_source)
        base_time_budget = self._extract_time_budget(base_source)

        configs = self._deriver.derive_configs(
            base_depth, base_aspect_ratio, base_head_dim, scales
        )

        results = []
        for cfg in configs:
            # Run baseline at this scale
            baseline_bpb = self._run_single(
                base_source, cfg, base_time_budget, is_modified=False
            )
            if baseline_bpb is None:
                results.append(self._make_failed_result(cfg, "baseline_failed"))
                continue

            # Apply modification and run
            modified_source = self._apply_diff(base_source, modification_diff)
            modified_bpb = self._run_single(
                modified_source, cfg, base_time_budget, is_modified=True
            )
            if modified_bpb is None:
                results.append(self._make_failed_result(cfg, "modified_failed"))
                # Early exit on failure at smallest scale
                if cfg.scale_factor == min(scales):
                    break
                continue

            delta = modified_bpb - baseline_bpb  # negative = improvement

            result = ScalingResult(
                scale_factor=cfg.scale_factor,
                config=asdict(cfg),
                val_bpb=modified_bpb,
                delta_vs_baseline=delta,
                training_seconds=0.0,
                converged=True,
            )
            results.append(result)

            # Early exit: if smallest scale shows regression, skip larger scales
            if cfg.scale_factor == min(scales) and delta > 0:
                break

        return results

    def _run_single(
        self,
        source: str,
        config: ScaleConfig,
        base_time_budget: int,
        is_modified: bool,
    ) -> float | None:
        """Run training at a given scale and return val_bpb, or None on failure."""
        # Scale time budget proportionally
        time_budget = max(30, int(base_time_budget * config.scale_factor))

        # Patch hyperparameters
        patched = self._patch_source(source, config, time_budget)

        # Write temp file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="train_scale_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(patched)

            # Run via subprocess
            timeout = min(MAX_RUN_TIMEOUT, time_budget * 3 + 120)
            try:
                result = subprocess.run(
                    ["uv", "run", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.path.dirname(os.path.abspath(tmp_path)),
                )
            except subprocess.TimeoutExpired:
                return None
            except OSError:
                return None

            if result.returncode != 0:
                return None

            return self._parse_val_bpb(result.stdout)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _patch_source(self, source: str, config: ScaleConfig, time_budget: int) -> str:
        """Patch DEPTH, DEVICE_BATCH_SIZE, TIME_BUDGET in the source."""
        patched = source
        patched = _HP_PATTERNS["DEPTH"].sub(
            rf"\g<1>{config.depth}", patched
        )
        # Scale batch size down for smaller models to avoid wasting memory
        # but keep it reasonable
        base_batch = 128
        scaled_batch = max(16, min(base_batch, int(base_batch * max(0.5, config.scale_factor))))
        patched = _HP_PATTERNS["DEVICE_BATCH_SIZE"].sub(
            rf"\g<1>{scaled_batch}", patched
        )
        # Patch TIME_BUDGET in prepare.py import section — it's defined there
        # but also need to handle it in the source
        patched = _HP_PATTERNS["TIME_BUDGET"].sub(
            rf"\g<1>{time_budget}", patched
        )
        return patched

    def _apply_diff(self, base_source: str, diff: str) -> str:
        """Apply a modification diff to the source.

        The diff can be:
        - A unified diff (lines starting with +/-)
        - A direct replacement dict as string
        - Raw source code (returned as-is if it looks like valid Python)
        """
        if not diff or not diff.strip():
            return base_source

        # If the diff looks like complete Python source, use it directly
        lines = diff.strip().split("\n")
        if any(line.startswith("import ") or line.startswith("from ") for line in lines[:10]):
            if len(lines) > 20:
                return diff

        # Try to apply as unified diff
        result = base_source
        additions = []
        removals = []
        for line in lines:
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                continue
            if line.startswith("-") and not line.startswith("---"):
                removals.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                additions.append(line[1:])

        if removals or additions:
            for removal in removals:
                result = result.replace(removal, "", 1)
            # Insert additions at the point of first removal, or at end
            if additions:
                insert_text = "\n".join(additions)
                if removals:
                    # Find where the first removal was and insert additions nearby
                    idx = base_source.find(removals[0])
                    if idx >= 0:
                        result = result[:idx] + insert_text + "\n" + result[idx:]
                    else:
                        result += "\n" + insert_text
                else:
                    result += "\n" + insert_text

        return result

    def _extract_hp(self, source: str) -> tuple[int, int, int]:
        """Extract DEPTH, ASPECT_RATIO, HEAD_DIM from source."""
        depth = self._extract_int(source, "DEPTH")
        aspect = self._extract_int(source, "ASPECT_RATIO")
        head_dim = self._extract_int(source, "HEAD_DIM")
        return depth, aspect, head_dim

    def _extract_time_budget(self, source: str) -> int:
        m = re.search(r"TIME_BUDGET\s*=\s*(\d+)", source)
        return int(m.group(1)) if m else 300

    def _extract_int(self, source: str, name: str) -> int:
        m = re.search(rf"^{name}\s*=\s*(\d+)", source, re.MULTILINE)
        if m:
            return int(m.group(1))
        raise ValueError(f"Could not find {name} in source")

    def _parse_val_bpb(self, stdout: str) -> float | None:
        """Parse val_bpb from training script output."""
        m = _VAL_BPB_RE.search(stdout)
        if m:
            val = float(m.group(1))
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        return None

    def _read_file(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    def _make_failed_result(self, config: ScaleConfig, reason: str) -> ScalingResult:
        return ScalingResult(
            scale_factor=config.scale_factor,
            config=asdict(config),
            val_bpb=0.0,
            delta_vs_baseline=0.0,
            training_seconds=0.0,
            converged=False,
        )
