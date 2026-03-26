"""
Main bandit loop: orchestrates one iteration of the adaptive bandit pipeline.
"""

import logging

logger = logging.getLogger(__name__)

import time
import copy

from .schemas import (
    BanditState, ArmState, LoopContext, IterationResult,
    SelectionResult, DispatchContext, DispatchResult,
    AcceptanceDecision, ReheatEvent, RollbackResult, HealthAlert,
    save_jsonl,
)


class BanditLoop:
    """Orchestrates a single bandit iteration through all pipeline stages."""

    def __init__(self):
        self._sampler = None
        self._booster = None
        self._dispatcher = None
        self._acceptance = None
        self._posterior = None
        self._reheat = None
        self._rollback = None
        self._health = None
        self._regime = None
        self._temp_deriver = None

    # ------------------------------------------------------------------
    # Lazy imports to avoid circular dependencies with Phase 2/3 modules
    # ------------------------------------------------------------------

    def _get_sampler(self):
        if self._sampler is None:
            try:
                from .sampler import ThompsonSamplerEngine
                self._sampler = ThompsonSamplerEngine()
            except ImportError:
                self._sampler = None
        return self._sampler

    def _get_booster(self):
        if self._booster is None:
            try:
                from .boosting import DiagnosticsArmBooster
                self._booster = DiagnosticsArmBooster()
            except ImportError:
                self._booster = None
        return self._booster

    def _get_dispatcher(self):
        if self._dispatcher is None:
            try:
                from .dispatch import BanditDispatchRouter
                self._dispatcher = BanditDispatchRouter()
            except ImportError:
                self._dispatcher = None
        return self._dispatcher

    def _get_acceptance(self):
        if self._acceptance is None:
            try:
                from .acceptance import AnnealingAcceptanceEngine
                self._acceptance = AnnealingAcceptanceEngine()
            except ImportError:
                self._acceptance = None
        return self._acceptance

    def _get_posterior(self):
        if self._posterior is None:
            try:
                from .updater import PosteriorUpdateEngine
                self._posterior = PosteriorUpdateEngine()
            except ImportError:
                self._posterior = None
        return self._posterior

    def _get_reheat(self):
        if self._reheat is None:
            try:
                from .reheat import AdaptiveReheatEngine
                self._reheat = AdaptiveReheatEngine()
            except ImportError:
                self._reheat = None
        return self._reheat

    def _get_rollback(self):
        if self._rollback is None:
            try:
                from .safety import RollbackSafetyNet
                self._rollback = RollbackSafetyNet()
            except ImportError:
                self._rollback = None
        return self._rollback

    def _get_health(self):
        if self._health is None:
            try:
                from .health import PosteriorHealthChecker
                self._health = PosteriorHealthChecker()
            except ImportError:
                self._health = None
        return self._health

    def _get_regime(self):
        if self._regime is None:
            try:
                from .regime import RegimeTransitionManager
                self._regime = RegimeTransitionManager()
            except ImportError:
                self._regime = None
        return self._regime

    def _get_temp_deriver(self):
        if self._temp_deriver is None:
            try:
                from .temperature import TemperatureDeriver
                self._temp_deriver = TemperatureDeriver()
            except ImportError:
                self._temp_deriver = None
        return self._temp_deriver

    # ------------------------------------------------------------------
    # Fallback for no_bandit regime (delegate to EvaluationScheduler)
    # ------------------------------------------------------------------

    def _fallback_iteration(self, context: LoopContext, state: BanditState) -> IterationResult:
        """Run a fallback iteration when in no_bandit regime."""
        return IterationResult(
            iteration=state.global_iteration,
            arm_selected="",
            dispatch_path="scheduler_fallback",
            verdict="no_bandit_regime",
            accepted=False,
            accepted_by="scheduler_fallback",
        )

    # ------------------------------------------------------------------
    # Main iteration
    # ------------------------------------------------------------------

    def run_iteration(self, context: LoopContext) -> IterationResult:
        """Execute one full bandit iteration.

        Steps:
        1. Regime transition check
        2. If no_bandit: fallback
        3. Diagnostics boost compute + apply
        4. Thompson Sampling selection
        5. Log selection
        6. Dispatch
        7. Annealing acceptance
        8. Posterior update
        9. Adaptive reheat check
        10. Rollback safety net
        11. Posterior health check
        12. Decay boosts
        """
        t_start = time.time()
        state = context.bandit_state
        if state is None:
            state = BanditState()

        rng = context.rng
        regime_at_start = state.regime

        # 1. Regime transition check
        regime_mgr = self._get_regime()
        if regime_mgr is not None:
            try:
                _, next_regime = regime_mgr.check_transition(state, context.journal_path)
                if next_regime:
                    state.regime = next_regime
            except Exception as e:
                logger.exception(e)  # Continue under current regime

        # 2. If no_bandit, delegate to scheduler fallback
        if state.regime == "no_bandit":
            result = self._fallback_iteration(context, state)
            result.elapsed_seconds = time.time() - t_start
            context.bandit_state = state
            return result

        # 3. Diagnostics boost
        booster = self._get_booster()
        boosted_state = state
        if booster is not None:
            try:
                boosts = booster.compute_boosts(
                    context.diagnostics_report, None, state)
                boosted_state = booster.apply_boosts(state, boosts)
            except Exception:
                boosted_state = state

        # 4. Thompson Sampling selection
        sampler = self._get_sampler()
        if sampler is None:
            result = self._fallback_iteration(context, state)
            result.elapsed_seconds = time.time() - t_start
            context.bandit_state = state
            return result

        selection = sampler.select(boosted_state, rng=rng)

        # Handle empty selection
        if not selection.arm_id:
            result = self._fallback_iteration(context, state)
            result.elapsed_seconds = time.time() - t_start
            context.bandit_state = state
            return result

        # 5. Log selection
        if context.log_writer is not None:
            try:
                save_jsonl({
                    "type": "selection",
                    "iteration": state.global_iteration,
                    "arm_id": selection.arm_id,
                    "selected_by": selection.selected_by,
                    "dispatch_path": selection.dispatch_path,
                    "timestamp": time.time(),
                }, getattr(context.log_writer, "path", "bandit_log.jsonl"))
            except Exception as e:
                logger.exception(e)

        # 6. Dispatch
        dispatch_result = None
        dispatcher = self._get_dispatcher()
        if dispatcher is not None:
            dispatch_ctx = DispatchContext(
                model_scientist_pipeline=context.model_scientist_pipeline,
                surrogate_triage_pipeline=context.surrogate_triage_pipeline,
                gpu_kernel_pipeline=context.gpu_kernel_pipeline,
                queue_manager=context.queue_manager,
                base_source=context.base_source,
                diagnostics_report=context.diagnostics_report,
            )
            try:
                dispatch_result = dispatcher.dispatch(selection, state, dispatch_ctx)
            except Exception as e:
                # Dispatch exception -> treat as failure
                dispatch_result = DispatchResult(
                    arm_id=selection.arm_id,
                    dispatch_path=selection.dispatch_path,
                    success=False,
                    delta=None,
                    verdict="dispatch_error",
                    error=str(e),
                )

        # Handle dispatch returning None -> resample once
        if dispatch_result is None:
            selection = sampler.select(boosted_state, rng=rng)
            if selection.arm_id and dispatcher is not None:
                dispatch_ctx = DispatchContext(
                    model_scientist_pipeline=context.model_scientist_pipeline,
                    surrogate_triage_pipeline=context.surrogate_triage_pipeline,
                    gpu_kernel_pipeline=context.gpu_kernel_pipeline,
                    queue_manager=context.queue_manager,
                    base_source=context.base_source,
                    diagnostics_report=context.diagnostics_report,
                )
                try:
                    dispatch_result = dispatcher.dispatch(selection, state, dispatch_ctx)
                except Exception:
                    dispatch_result = DispatchResult(
                        arm_id=selection.arm_id,
                        dispatch_path=selection.dispatch_path,
                        success=False,
                        delta=None,
                        verdict="dispatch_error_resample",
                    )

        # If still no result, create a failure stub
        if dispatch_result is None:
            dispatch_result = DispatchResult(
                arm_id=selection.arm_id,
                dispatch_path=selection.dispatch_path,
                success=False,
                delta=None,
                verdict="no_dispatch",
            )

        # 7. Annealing acceptance decision
        acceptance = AcceptanceDecision()
        acceptance_engine = self._get_acceptance()
        if acceptance_engine is not None and dispatch_result.delta is not None:
            try:
                arm_state = state.arms.get(selection.arm_id)
                if isinstance(arm_state, ArmState):
                    acceptance = acceptance_engine.decide(
                        dispatch_result.delta, arm_state, state, rng=rng)
            except Exception:
                acceptance = AcceptanceDecision(
                    accepted=dispatch_result.delta is not None and dispatch_result.delta <= 0,
                    accepted_by="improvement" if (dispatch_result.delta is not None and dispatch_result.delta <= 0) else "rejected",
                )
        elif dispatch_result.delta is not None and dispatch_result.delta <= 0:
            acceptance = AcceptanceDecision(
                accepted=True, accepted_by="improvement")

        # 8. Posterior update
        posterior_engine = self._get_posterior()
        if posterior_engine is not None:
            try:
                state = posterior_engine.update(
                    state, dispatch_result, context.log_writer)
            except Exception:
                # Manual minimal update
                arm = state.arms.get(selection.arm_id)
                if isinstance(arm, ArmState):
                    arm.total_attempts += 1
                    if dispatch_result.success:
                        arm.total_successes += 1
                        arm.alpha += 1
                        arm.consecutive_failures = 0
                    else:
                        arm.beta += 1
                        arm.consecutive_failures += 1
                    arm.last_selected = time.time()
        else:
            # Minimal posterior update without engine
            arm = state.arms.get(selection.arm_id)
            if isinstance(arm, ArmState):
                arm.total_attempts += 1
                if dispatch_result.success:
                    arm.total_successes += 1
                    arm.alpha += 1
                    arm.consecutive_failures = 0
                else:
                    arm.beta += 1
                    arm.consecutive_failures += 1
                arm.last_selected = time.time()

        # 9. Adaptive reheat check
        reheat_triggered = False
        reheat_engine = self._get_reheat()
        if reheat_engine is not None:
            try:
                reheat_result = reheat_engine.check_and_reheat(state)
                if reheat_result is not None:
                    state = reheat_result if isinstance(reheat_result, BanditState) else state
                    reheat_triggered = True
            except Exception as e:
                logger.exception(e)

        # 10. Rollback safety net
        rollback_triggered = False
        rollback_net = self._get_rollback()
        if rollback_net is not None and state.enable_rollback_safety:
            try:
                rollback_result = rollback_net.check_and_rollback(state, dispatch_result)
                if rollback_result is not None and hasattr(rollback_result, "rolled_back"):
                    rollback_triggered = rollback_result.rolled_back
            except Exception as e:
                logger.exception(e)

        # 11. Posterior health check
        health_alerts = []
        health_checker = self._get_health()
        if health_checker is not None:
            try:
                alerts = health_checker.check(state)
                if alerts:
                    health_alerts = [a.to_dict() if hasattr(a, "to_dict") else a
                                     for a in alerts]
            except Exception as e:
                logger.exception(e)

        # 12. Decay boosts
        if booster is not None:
            try:
                state = booster.decay_all_boosts(state)
            except Exception as e:
                logger.exception(e)

        context.bandit_state = state

        elapsed = time.time() - t_start
        return IterationResult(
            iteration=context.bandit_state.global_iteration - 1 if self._get_posterior() is not None else context.bandit_state.global_iteration,
            arm_selected=selection.arm_id,
            dispatch_path=dispatch_result.dispatch_path,
            delta=dispatch_result.delta,
            verdict=dispatch_result.verdict,
            accepted=acceptance.accepted,
            accepted_by=acceptance.accepted_by,
            temperature=acceptance.T_effective if hasattr(acceptance, "T_effective") else 0.0,
            reheat_triggered=reheat_triggered,
            rollback_triggered=rollback_triggered,
            health_alerts=health_alerts,
            elapsed_seconds=elapsed,
        )
