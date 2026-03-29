"""
Bandit dispatch router — routes selected arms to appropriate pipelines.
"""

import time

from bandit.schemas import (
    BanditState, SelectionResult, DispatchContext, DispatchResult,
)


class BanditDispatchRouter:
    """Routes arm selections to the appropriate pipeline for evaluation."""

    def dispatch(self, selection: SelectionResult, state: BanditState,
                 context: DispatchContext) -> DispatchResult:
        """Dispatch the selected arm to its pipeline.

        Routes based on dispatch_path:
        - internal: ModelScientistPipeline.evaluate_modification
        - paper: queue pop + evaluate
        - kernel: GPUKernelPipeline methods

        On exception: returns DispatchResult(success=False, error=str(e)).
        """
        start = time.time()
        try:
            if selection.dispatch_path == "paper":
                return self._dispatch_paper(selection, state, context, start)
            elif selection.dispatch_path == "kernel":
                return self._dispatch_kernel(selection, state, context, start)
            else:
                return self._dispatch_internal(selection, state, context, start)
        except Exception as e:
            return DispatchResult(
                arm_id=selection.arm_id,
                dispatch_path=selection.dispatch_path,
                success=False,
                error=str(e),
                elapsed_seconds=time.time() - start,
            )

    def _dispatch_internal(self, selection, state, context, start):
        """Dispatch to ModelScientistPipeline for internal modifications."""
        pipeline = context.model_scientist_pipeline
        if pipeline is None:
            return DispatchResult(
                arm_id=selection.arm_id,
                dispatch_path="internal",
                success=False,
                error="model_scientist_pipeline not available",
                elapsed_seconds=time.time() - start,
            )

        result = pipeline.evaluate_modification(
            arm_id=selection.arm_id,
            base_source=context.base_source,
            diagnostics_report=context.diagnostics_report,
        )

        return DispatchResult(
            arm_id=selection.arm_id,
            dispatch_path="internal",
            success=result.get("success", False) if isinstance(result, dict) else getattr(result, "success", False),
            delta=result.get("delta") if isinstance(result, dict) else getattr(result, "delta", None),
            verdict=result.get("verdict", "") if isinstance(result, dict) else getattr(result, "verdict", ""),
            journal_entry_id=result.get("journal_entry_id", "") if isinstance(result, dict) else getattr(result, "journal_entry_id", ""),
            elapsed_seconds=time.time() - start,
        )

    def _dispatch_paper(self, selection, state, context, start):
        """Dispatch paper-sourced modification via queue pop + evaluate."""
        from bandit.queue_bridge import QueueFilteredPopper

        popper = QueueFilteredPopper()
        entry = popper.pop_matching(context.queue_manager, selection.arm_id, source_type="paper")

        if entry is None:
            return DispatchResult(
                arm_id=selection.arm_id,
                dispatch_path="paper",
                success=False,
                error="no matching queue entry",
                elapsed_seconds=time.time() - start,
            )

        pipeline = context.model_scientist_pipeline
        if pipeline is None:
            return DispatchResult(
                arm_id=selection.arm_id,
                dispatch_path="paper",
                success=False,
                error="model_scientist_pipeline not available",
                elapsed_seconds=time.time() - start,
            )

        result = pipeline.evaluate_modification(
            modified_source=context.base_source,  # applying diff would be ideal, mock for now
            hypothesis=f"Paper technique: {selection.arm_id}",
            predicted_delta=-0.02,
            tags=[selection.arm_id, "paper"],
        )

        return DispatchResult(
            arm_id=selection.arm_id,
            dispatch_path="paper",
            success=result.get("success", False) if isinstance(result, dict) else getattr(result, "success", False),
            delta=result.get("delta") if isinstance(result, dict) else getattr(result, "delta", None),
            verdict=result.get("verdict", "") if isinstance(result, dict) else getattr(result, "verdict", ""),
            journal_entry_id=result.get("journal_entry_id", "") if isinstance(result, dict) else getattr(result, "journal_entry_id", ""),
            elapsed_seconds=time.time() - start,
        )

    def _dispatch_kernel(self, selection, state, context, start):
        """Dispatch to GPUKernelPipeline for kernel discovery/evolution."""
        pipeline = context.gpu_kernel_pipeline
        if pipeline is None:
            return DispatchResult(
                arm_id=selection.arm_id,
                dispatch_path="kernel",
                success=False,
                error="gpu_kernel_pipeline not available",
                elapsed_seconds=time.time() - start,
            )

        # Determine if this is discovery or evolution based on arm_id
        if "evolution" in selection.arm_id:
            result = pipeline.evolve_kernel(arm_id=selection.arm_id)
        else:
            result = pipeline.discover_kernel(arm_id=selection.arm_id)

        return DispatchResult(
            arm_id=selection.arm_id,
            dispatch_path="kernel",
            success=result.get("success", False) if isinstance(result, dict) else getattr(result, "success", False),
            delta=result.get("delta") if isinstance(result, dict) else getattr(result, "delta", None),
            verdict=result.get("verdict", "") if isinstance(result, dict) else getattr(result, "verdict", ""),
            journal_entry_id=result.get("journal_entry_id", "") if isinstance(result, dict) else getattr(result, "journal_entry_id", ""),
            elapsed_seconds=time.time() - start,
        )
