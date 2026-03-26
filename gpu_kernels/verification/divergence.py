"""
Training divergence detection for generated kernels.
"""

import traceback
import copy

import torch
import torch.nn as nn

from gpu_kernels.schemas import DivergenceResult


class TrainingDivergenceDetector:
    """Detect whether a kernel causes training divergence compared to a reference."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check(
        self,
        kernel_callable,
        reference_callable,
        train_config: dict,
        n_steps: int = 100,
        seed: int = 42,
    ) -> DivergenceResult:
        """Run parallel short training sequences and compare trajectories.

        Args:
            kernel_callable: The kernel function (will replace reference in the model forward).
            reference_callable: The reference PyTorch function.
            train_config: Dict with keys:
                - model_fn: callable() -> nn.Module (factory to create model)
                - make_batch_fn: callable(step, device) -> (x, y) batch generator
                - loss_fn: callable(model, x, y) -> loss (optional, defaults to model(x, y))
                - patch_fn: callable(model, kernel_callable) -> model
                    (patches the model to use the kernel instead of reference)
                - lr: float (learning rate, default 1e-3)
            n_steps: Number of training steps to run.
            seed: Random seed for reproducibility.

        Returns:
            DivergenceResult comparing the two training runs.
        """
        if not torch.cuda.is_available():
            return DivergenceResult()

        result = DivergenceResult()

        try:
            model_fn = train_config["model_fn"]
            make_batch_fn = train_config["make_batch_fn"]
            loss_fn = train_config.get("loss_fn", None)
            patch_fn = train_config.get("patch_fn", None)
            lr = train_config.get("lr", 1e-3)

            # Run reference training
            ref_losses, ref_grad_norms, ref_params = self._run_training(
                model_fn, make_batch_fn, loss_fn, None, None, lr, n_steps, seed,
            )

            # Run kernel training (patched model)
            kern_losses, kern_grad_norms, kern_params = self._run_training(
                model_fn, make_batch_fn, loss_fn, patch_fn, kernel_callable, lr, n_steps, seed,
            )

            result.loss_curve_reference = ref_losses
            result.loss_curve_kernel = kern_losses

            # Compare losses
            if ref_losses and kern_losses:
                loss_diffs = [abs(r - k) for r, k in zip(ref_losses, kern_losses)]
                mean_abs_loss = sum(abs(l) for l in ref_losses) / len(ref_losses)
                threshold = 0.001 * mean_abs_loss if mean_abs_loss > 0 else 1e-6
                result.max_loss_divergence = max(loss_diffs)

                diverged_step = None
                for i, d in enumerate(loss_diffs):
                    if d > threshold:
                        diverged_step = i
                        break
                result.diverged_at_step = diverged_step

            # Compare gradient norms (sampled every 10 steps)
            if ref_grad_norms and kern_grad_norms:
                grad_diffs = [abs(r - k) for r, k in zip(ref_grad_norms, kern_grad_norms)]
                result.max_grad_norm_divergence = max(grad_diffs) if grad_diffs else 0.0

            # Compare final parameters
            if ref_params is not None and kern_params is not None:
                result.final_param_match = self._params_match(ref_params, kern_params)

            # Verdict
            result.passed = (
                result.diverged_at_step is None
                and result.final_param_match
            )

        except Exception:
            result.passed = False
            result.loss_curve_reference = []
            result.loss_curve_kernel = []

        return result

    def _run_training(
        self, model_fn, make_batch_fn, loss_fn, patch_fn, kernel_callable, lr, n_steps, seed,
    ):
        """Run a short training loop and collect metrics."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        model = model_fn().to(self.device)
        if patch_fn is not None and kernel_callable is not None:
            model = patch_fn(model, kernel_callable)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        grad_norms = []

        for step in range(n_steps):
            x, y = make_batch_fn(step, self.device)
            optimizer.zero_grad()

            if loss_fn is not None:
                loss = loss_fn(model, x, y)
            else:
                loss = model(x, y)

            loss.backward()

            # Record gradient norm every 10 steps
            if step % 10 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.float().norm().item() ** 2
                grad_norms.append(total_norm ** 0.5)

            optimizer.step()
            losses.append(loss.item())

        # Collect final parameters
        final_params = {name: p.detach().clone() for name, p in model.named_parameters()}

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()

        return losses, grad_norms, final_params

    def _params_match(self, ref_params, kern_params, rtol=1e-4, atol=1e-5):
        """Check if final parameters are close enough."""
        for name in ref_params:
            if name not in kern_params:
                return False
            if not torch.allclose(
                ref_params[name].float(), kern_params[name].float(),
                rtol=rtol, atol=atol,
            ):
                return False
        return True
