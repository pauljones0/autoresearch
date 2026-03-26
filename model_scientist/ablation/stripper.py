"""
Phase 3 — ComponentStripper: After ablation, strip neutral/negative components
and produce a cleaned train.py with only beneficial changes.
"""

import os
import re
import subprocess
import tempfile

from ..schemas import AblationReport
from model_scientist.ablation.component_isolator import ComponentIsolator


_VAL_BPB_RE = re.compile(r'val_bpb:\s+([\d.]+)')

DEFAULT_TIMEOUT = 1800


class ComponentStripper:
    """Strip non-beneficial components from a modification after ablation."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT,
                 python_cmd: str = "uv run"):
        self.timeout = timeout
        self.python_cmd = python_cmd
        self.isolator = ComponentIsolator()

    def strip(self, base_source: str,
              ablation_report: AblationReport) -> tuple:
        """Strip components with marginal_contribution <= 0.

        Args:
            base_source: Original unmodified train.py source.
            ablation_report: Completed ablation report.

        Returns:
            (stripped_source: str, stripped_component_ids: list[int])
        """
        if not ablation_report.ablation_results:
            # No ablation was run — nothing to strip
            return base_source, []

        components = ablation_report.components
        results = ablation_report.ablation_results

        # Build lookup of marginal contributions
        marginal_map = {}
        for result in results:
            marginal_map[result.component_id] = result.marginal_contribution

        # Identify which components to keep and which to strip
        keep_indices = []
        stripped_ids = []

        for idx, component in enumerate(components):
            cid = component.component_id if hasattr(component, 'component_id') else component.get('component_id', idx)
            marginal = marginal_map.get(cid, None)

            if marginal is not None and marginal <= 0:
                stripped_ids.append(cid)
            else:
                keep_indices.append(idx)

        if not stripped_ids:
            # Nothing to strip — all components are beneficial
            # Reconstruct full modification from all components
            return self.isolator.apply_subset(
                base_source, components, list(range(len(components)))
            ), []

        if not keep_indices:
            # All components are neutral/negative — revert to base
            return base_source, stripped_ids

        # Apply only the beneficial components
        stripped_source = self.isolator.apply_subset(
            base_source, components, keep_indices)

        # Update the report
        ablation_report.stripped_components = stripped_ids

        return stripped_source, stripped_ids

    def evaluate_stripped(self, stripped_source: str,
                          original_val_bpb: float) -> dict:
        """Evaluate the stripped version and compare to original.

        Args:
            stripped_source: The stripped train.py source.
            original_val_bpb: val_bpb of the full (unstripped) modification.

        Returns:
            Dict with stripped_val_bpb, original_val_bpb, delta, improved.
        """
        stripped_val_bpb = self._run_variant(stripped_source)

        if stripped_val_bpb is None:
            return {
                "stripped_val_bpb": None,
                "original_val_bpb": original_val_bpb,
                "delta": None,
                "improved": False,
                "error": "Stripped variant failed to run",
            }

        delta = original_val_bpb - stripped_val_bpb  # positive = stripped is better
        return {
            "stripped_val_bpb": stripped_val_bpb,
            "original_val_bpb": original_val_bpb,
            "delta": delta,
            "improved": delta > 0,
        }

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _run_variant(self, source: str) -> float:
        """Write source to a temp file, run it, parse val_bpb.

        Returns val_bpb float or None on failure.
        """
        tmp_dir = tempfile.mkdtemp(prefix="ablation_strip_")
        tmp_path = os.path.join(tmp_dir, "train_stripped.py")

        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(source)

            cmd = f"{self.python_cmd} {tmp_path}"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=os.path.dirname(os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', '..'))),
            )

            if result.returncode != 0:
                return None

            output = result.stdout + result.stderr
            match = _VAL_BPB_RE.search(output)
            if match:
                return float(match.group(1))
            return None

        except subprocess.TimeoutExpired:
            return None
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass
