"""
ReferenceCodeExtractor: extracts standalone PyTorch reference functions
from train.py source for each FuseableGroup, enabling correctness
comparison against generated Triton kernels.
"""

import os
import textwrap

from ..schemas import FuseableGroup


# Templates for reference implementations of known fuseable patterns
_REFERENCE_TEMPLATES = {
    "rmsnorm": textwrap.dedent("""\
        import torch
        import torch.nn.functional as F

        def reference_rmsnorm(x: torch.Tensor) -> torch.Tensor:
            \"\"\"RMS normalization: x / rms(x), matching F.rms_norm.\"\"\"
            return F.rms_norm(x, (x.size(-1),))
    """),
    "mlp_activation": textwrap.dedent("""\
        import torch
        import torch.nn.functional as F

        def reference_mlp_activation(x: torch.Tensor) -> torch.Tensor:
            \"\"\"MLP activation: relu(x)^2 (ReGLU-style squared ReLU).\"\"\"
            return F.relu(x).square()
    """),
    "softcap": textwrap.dedent("""\
        import torch

        def reference_softcap(logits: torch.Tensor, cap: float = 15.0) -> torch.Tensor:
            \"\"\"Soft capping: cap * tanh(logits / cap).\"\"\"
            return cap * torch.tanh(logits / cap)
    """),
    "ve_gate": textwrap.dedent("""\
        import torch

        def reference_ve_gate(
            v: torch.Tensor, ve: torch.Tensor, gate_logits: torch.Tensor
        ) -> torch.Tensor:
            \"\"\"Value embedding gating: v + 2*sigmoid(gate) * ve.\"\"\"
            gate = 2.0 * torch.sigmoid(gate_logits)
            return v + gate.unsqueeze(-1) * ve
    """),
    "rotary_emb": textwrap.dedent("""\
        import torch

        def reference_rotary_emb(
            x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
        ) -> torch.Tensor:
            \"\"\"Apply rotary positional embeddings.\"\"\"
            d = x.shape[3] // 2
            x1, x2 = x[..., :d], x[..., d:]
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat([y1, y2], 3)
    """),
    "residual_scale": textwrap.dedent("""\
        import torch

        def reference_residual_scale(
            x: torch.Tensor, x0: torch.Tensor,
            resid_lambda: torch.Tensor, x0_lambda: torch.Tensor
        ) -> torch.Tensor:
            \"\"\"Scaled residual: resid_lambda * x + x0_lambda * x0.\"\"\"
            return resid_lambda * x + x0_lambda * x0
    """),
    "cross_entropy": textwrap.dedent("""\
        import torch
        import torch.nn.functional as F

        def reference_cross_entropy(
            logits: torch.Tensor, targets: torch.Tensor
        ) -> torch.Tensor:
            \"\"\"Cross entropy loss with ignore_index=-1.\"\"\"
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='mean',
            )
    """),
    "elementwise": textwrap.dedent("""\
        import torch

        def reference_elementwise(x: torch.Tensor) -> torch.Tensor:
            \"\"\"Fused elementwise operations (placeholder — customize per group).\"\"\"
            # TODO: Replace with actual op sequence from profiling
            return x
    """),
}


class ReferenceCodeExtractor:
    """Extracts standalone PyTorch reference code for fuseable groups."""

    def extract(self, group: FuseableGroup, train_source: str = "") -> str:
        """Generate a standalone reference implementation for a fuseable group.

        Args:
            group: The FuseableGroup to generate reference code for.
            train_source: Source code of train.py (used for context).

        Returns:
            Python source code string for the reference implementation.
        """
        # Try to match a known template based on group_id prefix
        for pattern_name, template in _REFERENCE_TEMPLATES.items():
            if group.group_id.startswith(pattern_name):
                return self._customize_template(template, group)

        # Fallback: generate from op names
        return self._generate_from_ops(group)

    def extract_and_save(
        self,
        group: FuseableGroup,
        output_dir: str,
        train_source: str = "",
    ) -> str:
        """Extract reference code and save to disk.

        Args:
            group: The FuseableGroup.
            output_dir: Base directory for reference_catalog.
            train_source: Source of train.py.

        Returns:
            Path to the saved reference file.
        """
        code = self.extract(group, train_source)
        group_dir = os.path.join(output_dir, group.group_id)
        os.makedirs(group_dir, exist_ok=True)
        ref_path = os.path.join(group_dir, "reference.py")
        with open(ref_path, "w") as f:
            f.write(code)
        return ref_path

    def _customize_template(self, template: str, group: FuseableGroup) -> str:
        """Add group-specific metadata as comments to a template."""
        header = (
            f"# Reference implementation for group: {group.group_id}\n"
            f"# Fusion type: {group.fusion_type}\n"
            f"# Ops: {', '.join(group.op_names)}\n"
            f"# Estimated speedup: {group.estimated_speedup_ratio:.2f}x\n\n"
        )
        return header + template

    def _generate_from_ops(self, group: FuseableGroup) -> str:
        """Generate a placeholder reference from op names."""
        op_list = "\n".join(f"    # {op}" for op in group.op_names)
        return (
            f"# Reference implementation for group: {group.group_id}\n"
            f"# Fusion type: {group.fusion_type}\n"
            f"# Ops: {', '.join(group.op_names)}\n\n"
            f"import torch\n\n"
            f"def reference_{group.group_id.replace('-', '_')}(x: torch.Tensor) -> torch.Tensor:\n"
            f'    """Fused operation sequence:\n{op_list}\n    """\n'
            f"    # TODO: Implement actual op sequence\n"
            f"    return x\n"
        )
