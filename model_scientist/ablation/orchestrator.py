"""
Phase 3 — AblationOrchestrator: Run leave-one-out ablation experiments
by executing train.py variants via subprocess.
"""

import os
import re
import subprocess
import tempfile
import time

from ..schemas import (
    ModificationComponent, AblationResult, AblationReport,
)
from model_scientist.ablation.diff_parser import DiffParser
from model_scientist.ablation.component_isolator import ComponentIsolator
from model_scientist.ablation.budgeter import AblationBudgeter


# Regex to extract val_bpb from training output
_VAL_BPB_RE = re.compile(r'val_bpb:\s+([\d.]+)')

# Default timeout per variant run (seconds)
DEFAULT_TIMEOUT = 1800  # 30 minutes


class AblationOrchestrator:
    """Run leave-one-out ablation to measure marginal contribution of each component."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT,
                 python_cmd: str = "uv run"):
        self.timeout = timeout
        self.python_cmd = python_cmd
        self.parser = DiffParser()
        self.isolator = ComponentIsolator()
        self.budgeter = AblationBudgeter()

    def run_ablation(self, base_source: str, modified_source: str,
                     base_val_bpb: float,
                     modification_id: str = "",
                     compute_budget: float = 0.0) -> AblationReport:
        """Run full leave-one-out ablation for a modification.

        Args:
            base_source: Original train.py source.
            modified_source: Modified train.py source.
            base_val_bpb: Baseline val_bpb from unmodified train.py.
            modification_id: Identifier for the modification.
            compute_budget: Max compute budget in seconds (0 = unlimited).

        Returns:
            AblationReport with marginal contributions for each component.
        """
        # Decompose modification into components
        components = self.parser.parse(base_source, modified_source)

        report = AblationReport(
            modification_id=modification_id,
            baseline_val_bpb=base_val_bpb,
            components=components,
        )

        # Check if ablation is warranted
        full_improvement = 0.0  # will be set after running full modification

        if len(components) <= 1:
            # Single component — nothing to ablate
            full_val_bpb = self._run_variant(modified_source)
            if full_val_bpb is not None:
                report.full_modification_val_bpb = full_val_bpb
                report.full_improvement = base_val_bpb - full_val_bpb
                report.final_val_bpb = full_val_bpb
            return report

        # Step 1: Run full modification to get its val_bpb
        full_val_bpb = self._run_variant(modified_source)
        if full_val_bpb is None:
            report.full_modification_val_bpb = 0.0
            report.full_improvement = 0.0
            report.final_val_bpb = base_val_bpb
            return report

        full_improvement = base_val_bpb - full_val_bpb
        report.full_modification_val_bpb = full_val_bpb
        report.full_improvement = full_improvement

        # Check budget
        should_run, reason = self.budgeter.should_ablate(
            n_components=len(components),
            improvement=full_improvement,
            compute_used=0.0,
            compute_budget=compute_budget,
        )

        if not should_run:
            report.final_val_bpb = full_val_bpb
            return report

        # Step 2: Run leave-one-out for each component
        ablation_results = []
        compute_used = 0.0

        for idx, component in enumerate(components):
            if compute_budget > 0:
                max_v = self.budgeter.max_variants(
                    compute_budget - compute_used, self.timeout)
                if max_v <= 0:
                    break

            t_start = time.time()
            loo_source = self.isolator.leave_one_out(
                base_source, components, exclude_idx=idx)
            loo_val_bpb = self._run_variant(loo_source)
            t_elapsed = time.time() - t_start
            compute_used += t_elapsed

            if loo_val_bpb is None:
                # Variant crashed — treat as if component is essential
                marginal = full_improvement
            else:
                improvement_without = base_val_bpb - loo_val_bpb
                marginal = full_improvement - improvement_without

            result = AblationResult(
                component_id=component.component_id,
                component_description=component.description,
                val_bpb_without=loo_val_bpb if loo_val_bpb is not None else 0.0,
                marginal_contribution=marginal,
            )
            ablation_results.append(result)

        report.ablation_results = ablation_results
        report.final_val_bpb = full_val_bpb
        return report

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _run_variant(self, source: str) -> float:
        """Write source to a temp file, run it, and parse val_bpb from output.

        Returns val_bpb float, or None if the run failed/timed out.
        """
        tmp_dir = tempfile.mkdtemp(prefix="ablation_")
        tmp_path = os.path.join(tmp_dir, "train_variant.py")

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

            # Parse val_bpb from stdout
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
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass
