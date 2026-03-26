"""
Optimizer correctness validator: extended multi-step validation with tight tolerances.
"""

from typing import Dict, Any, Callable, Optional

import torch

import sys, os


class OptimizerCorrectnessValidator:
    """Validate fused optimizer kernels with extended multi-step checks.

    Runs 500 steps comparing fused kernel output against reference optimizer,
    with tighter tolerances at milestones:
        - Step 100: param divergence < 1e-4
        - Step 500: param divergence < 5e-4
    """

    def __init__(
        self,
        param_tol_step_100: float = 1e-4,
        param_tol_step_500: float = 5e-4,
        grad_norm_tol: float = 1e-3,
        loss_tol: float = 1e-3,
    ):
        self.param_tol_step_100 = param_tol_step_100
        self.param_tol_step_500 = param_tol_step_500
        self.grad_norm_tol = grad_norm_tol
        self.loss_tol = loss_tol

    def _create_test_params(
        self, n_params: int = 4, param_size: int = 1024, device: str = "cuda"
    ) -> list:
        """Create reproducible test parameters."""
        torch.manual_seed(42)
        return [
            torch.randn(param_size, device=device, dtype=torch.float32, requires_grad=False)
            for _ in range(n_params)
        ]

    def _generate_synthetic_grads(
        self, params: list, step: int
    ) -> list:
        """Generate deterministic synthetic gradients for testing."""
        torch.manual_seed(step * 1000 + 7)
        return [
            torch.randn_like(p) * 0.01 * (1.0 + 0.001 * step)
            for p in params
        ]

    def _run_reference_step(
        self,
        params: list,
        grads: list,
        states: list,
        step: int,
        lr: float,
        betas: tuple,
        eps: float,
        weight_decay: float,
    ) -> list:
        """Run one reference AdamW step (matching train.py's adamw_step_fused logic)."""
        beta1, beta2 = betas
        updated = []

        for p, grad, state in zip(params, grads, states):
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            # Weight decay
            p_new = p * (1.0 - lr * weight_decay)

            # Momentum updates
            exp_avg.lerp_(grad, 1.0 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1.0 - beta2)

            # Bias correction
            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step
            denom = (exp_avg_sq / bias2).sqrt() + eps
            step_size = lr / bias1

            p_new = p_new - step_size * (exp_avg / denom)
            updated.append(p_new)

        return updated

    def validate(
        self,
        fused_kernel: Callable,
        reference_optimizer: Optional[Callable] = None,
        n_steps: int = 500,
    ) -> Dict[str, Any]:
        """Run extended correctness validation of fused optimizer kernel.

        Args:
            fused_kernel: The fused optimizer step callable. Signature:
                fused_kernel(p, grad, exp_avg, exp_avg_sq, step, lr, beta1, beta2, eps, wd)
            reference_optimizer: Optional custom reference. If None, uses
                built-in AdamW reference matching train.py.
            n_steps: Number of validation steps (default 500).

        Returns:
            Dict with validation results including pass/fail, divergence metrics,
            and per-checkpoint details.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lr = 0.004
        betas = (0.8, 0.95)
        eps = 1e-10
        weight_decay = 0.0

        # Create two copies of parameters
        ref_params = self._create_test_params(device=device)
        fused_params = [p.clone() for p in ref_params]

        ref_states = [{} for _ in ref_params]
        fused_states = [
            {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
            for p in fused_params
        ]

        checkpoints = {}
        max_param_divergence = 0.0
        failed_at_step = None
        failed_metric = ""

        for step in range(1, n_steps + 1):
            grads = self._generate_synthetic_grads(ref_params, step)

            # Reference step
            ref_params = self._run_reference_step(
                ref_params, grads, ref_states, step, lr, betas, eps, weight_decay
            )

            # Fused kernel step
            for i, (p, grad, state) in enumerate(zip(fused_params, grads, fused_states)):
                fused_kernel(
                    p, grad, state["exp_avg"], state["exp_avg_sq"],
                    step, lr, betas[0], betas[1], eps, weight_decay
                )
                fused_params[i] = p  # in case kernel returns new tensor

            # Compute divergence
            param_divergences = []
            for rp, fp in zip(ref_params, fused_params):
                div = (rp - fp).abs().max().item()
                param_divergences.append(div)
            step_max_div = max(param_divergences)
            max_param_divergence = max(max_param_divergence, step_max_div)

            # Check tolerance at milestones
            if step == 100:
                passed_100 = step_max_div < self.param_tol_step_100
                checkpoints["step_100"] = {
                    "max_param_divergence": step_max_div,
                    "tolerance": self.param_tol_step_100,
                    "passed": passed_100,
                }
                if not passed_100 and failed_at_step is None:
                    failed_at_step = step
                    failed_metric = f"param_divergence={step_max_div:.2e} > {self.param_tol_step_100:.2e}"

            if step == n_steps:
                passed_final = step_max_div < self.param_tol_step_500
                checkpoints[f"step_{n_steps}"] = {
                    "max_param_divergence": step_max_div,
                    "tolerance": self.param_tol_step_500,
                    "passed": passed_final,
                }
                if not passed_final and failed_at_step is None:
                    failed_at_step = step
                    failed_metric = f"param_divergence={step_max_div:.2e} > {self.param_tol_step_500:.2e}"

        overall_passed = all(
            cp["passed"] for cp in checkpoints.values()
        )

        return {
            "passed": overall_passed,
            "n_steps": n_steps,
            "max_param_divergence": max_param_divergence,
            "checkpoints": checkpoints,
            "failed_at_step": failed_at_step,
            "failed_metric": failed_metric,
            "config": {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "param_tol_step_100": self.param_tol_step_100,
                "param_tol_step_500": self.param_tol_step_500,
            },
        }
