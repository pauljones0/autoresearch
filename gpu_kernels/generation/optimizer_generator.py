"""
Triton optimizer fusion generator: generates fused AdamW update kernels.
"""

import os
import hashlib
import textwrap
from typing import List

from ..schemas import GeneratedKernel


# Variant configurations for optimizer kernels
OPTIMIZER_VARIANT_CONFIGS = [
    (512, 4),
    (1024, 4),
    (2048, 8),
]


class TritonOptimizerFusionGenerator:
    """Generate fused AdamW update Triton kernels.

    Fuses the full AdamW update path: gradient -> momentum update ->
    bias correction -> weight decay -> parameter update into a single
    kernel pass, minimizing memory traffic.
    """

    def __init__(self, output_dir: str = ""):
        if not output_dir:
            output_dir = os.path.join(
                os.path.dirname(__file__), '..', 'generated'
            )
        self.output_dir = os.path.abspath(output_dir)

    def _make_kernel_id(self, variant_index: int) -> str:
        raw = f"adamw_fused_v{variant_index}"
        short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"optim_adamw_{variant_index}_{short_hash}"

    def _generate_kernel_source(
        self, block_size: int, num_warps: int, kernel_id: str
    ) -> str:
        """Generate a fused AdamW Triton kernel source."""
        func_name = f"kernel_{kernel_id}".replace("-", "_")
        wrapper_name = f"fused_adamw_step_{block_size}"

        return textwrap.dedent(f'''\
            """
            Fused AdamW optimizer step kernel.
            Performs weight decay, momentum update, bias correction, and parameter
            update in a single pass to minimize memory traffic.

            Replaces the sequence:
                p.mul_(1 - lr * wd)
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.lerp_(grad.square(), 1 - beta2)
                bias1 = 1 - beta1 ** step
                bias2 = 1 - beta2 ** step
                denom = (exp_avg_sq / bias2).sqrt() + eps
                step_size = lr / bias1
                p.add_(exp_avg / denom, alpha=-step_size)
            """
            import torch
            import triton
            import triton.language as tl


            @triton.jit
            def {func_name}(
                p_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr, out_p_ptr,
                out_exp_avg_ptr, out_exp_avg_sq_ptr,
                n_elements,
                lr, beta1, beta2, eps, weight_decay,
                bias_correction1, bias_correction2,
                BLOCK_SIZE: tl.constexpr,
            ):
                pid = tl.program_id(0)
                offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements

                # Load all tensors in one pass
                p = tl.load(p_ptr + offsets, mask=mask)
                grad = tl.load(grad_ptr + offsets, mask=mask)
                m = tl.load(exp_avg_ptr + offsets, mask=mask)
                v = tl.load(exp_avg_sq_ptr + offsets, mask=mask)

                # Weight decay
                p = p * (1.0 - lr * weight_decay)

                # Update first moment: m = beta1 * m + (1 - beta1) * grad
                m = beta1 * m + (1.0 - beta1) * grad

                # Update second moment: v = beta2 * v + (1 - beta2) * grad^2
                grad_sq = grad * grad
                v = beta2 * v + (1.0 - beta2) * grad_sq

                # Bias-corrected update
                denom = tl.sqrt(v / bias_correction2) + eps
                step_size = lr / bias_correction1
                p = p - step_size * (m / denom)

                # Store all results in one pass
                tl.store(out_p_ptr + offsets, p, mask=mask)
                tl.store(out_exp_avg_ptr + offsets, m, mask=mask)
                tl.store(out_exp_avg_sq_ptr + offsets, v, mask=mask)


            def {wrapper_name}(
                p: torch.Tensor,
                grad: torch.Tensor,
                exp_avg: torch.Tensor,
                exp_avg_sq: torch.Tensor,
                step: int,
                lr: float,
                beta1: float,
                beta2: float,
                eps: float,
                weight_decay: float,
            ) -> None:
                """Fused AdamW step. Updates p, exp_avg, exp_avg_sq in-place."""
                assert p.is_cuda, "Parameters must be on CUDA"
                n_elements = p.numel()

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
                {func_name}[grid](
                    p, grad, exp_avg, exp_avg_sq,
                    p, exp_avg, exp_avg_sq,  # in-place output
                    n_elements,
                    lr, beta1, beta2, eps, weight_decay,
                    bias_correction1, bias_correction2,
                    BLOCK_SIZE={block_size},
                    num_warps={num_warps},
                )


            # Entry point for integration
            fused_adamw_step = {wrapper_name}
        ''')

    def generate(self) -> List[GeneratedKernel]:
        """Generate 3 fused AdamW kernel variants with different configurations.

        Returns:
            List of 3 GeneratedKernel objects.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        variants = []
        for idx, (block_size, num_warps) in enumerate(OPTIMIZER_VARIANT_CONFIGS):
            kernel_id = self._make_kernel_id(idx)
            source = self._generate_kernel_source(block_size, num_warps, kernel_id)

            kernel_path = os.path.join(self.output_dir, f"{kernel_id}.py")
            with open(kernel_path, 'w') as f:
                f.write(source)

            integration_diff = textwrap.dedent(f"""\
                # Integration diff for {kernel_id}
                # In MuonAdamW._step_adamw(), replace adamw_step_fused() call with:
                #     from gpu_kernels.active.{kernel_id} import fused_adamw_step
                #     fused_adamw_step(p, grad, state['exp_avg'], state['exp_avg_sq'],
                #                      state['step'], group['lr'], group['betas'][0],
                #                      group['betas'][1], group['eps'], group['weight_decay'])
            """)

            variant = GeneratedKernel(
                kernel_id=kernel_id,
                group_id="adamw_update",
                variant_index=idx,
                kernel_path=kernel_path,
                integration_diff=integration_diff,
                block_size=block_size,
                num_warps=num_warps,
                memory_strategy="row_major",
                fusion_type="optimizer",
            )
            variants.append(variant)

        return variants
