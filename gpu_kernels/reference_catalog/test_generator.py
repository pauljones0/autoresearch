"""
TestInputGenerator: generates 5 test configurations with expected outputs
for correctness verification of generated Triton kernels against PyTorch
reference implementations.
"""

import os
from typing import Callable

import torch



class TestInputGenerator:
    """Generates test input configurations and expected outputs.

    Produces 5 configs:
      - minimal: smallest valid shape for fast smoke tests
      - typical: production-like shapes from model config
      - stress: larger shapes to test performance and memory
      - edge_zeros: all-zero inputs to test zero-handling
      - edge_extremes: inputs with very large/small values
    """

    def generate(
        self,
        reference_callable: Callable,
        shapes: dict,
        model_config: dict | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Generate test configs with inputs and expected outputs.

        Args:
            reference_callable: The reference function to generate expected outputs.
            shapes: Shape specification dict from TensorShapeDocumenter.
            model_config: GPTConfig as dict for deriving typical shapes.
            device: Device for tensors.
            dtype: Default dtype for test tensors.

        Returns:
            Dict mapping config_name to (inputs, expected_outputs) tuples.
        """
        # Extract representative shape from shapes spec
        input_shapes = self._extract_input_shapes(shapes)
        if not input_shapes:
            # Fallback to a reasonable default
            input_shapes = [[2, 64, 768]]

        configs = {}

        # 1. minimal: smallest valid shape
        minimal_inputs = self._make_inputs(
            input_shapes, scale_factor=0.25, min_dim=1, device=device, dtype=dtype
        )
        configs["minimal"] = self._run_reference(reference_callable, minimal_inputs)

        # 2. typical: production shapes
        typical_inputs = self._make_inputs(
            input_shapes, scale_factor=1.0, min_dim=1, device=device, dtype=dtype
        )
        configs["typical"] = self._run_reference(reference_callable, typical_inputs)

        # 3. stress: 2x production shapes
        stress_inputs = self._make_inputs(
            input_shapes, scale_factor=2.0, min_dim=1, device=device, dtype=dtype
        )
        configs["stress"] = self._run_reference(reference_callable, stress_inputs)

        # 4. edge_zeros: all zeros
        zero_inputs = self._make_zero_inputs(
            input_shapes, device=device, dtype=dtype
        )
        configs["edge_zeros"] = self._run_reference(reference_callable, zero_inputs)

        # 5. edge_extremes: extreme values
        extreme_inputs = self._make_extreme_inputs(
            input_shapes, device=device, dtype=dtype
        )
        configs["edge_extremes"] = self._run_reference(
            reference_callable, extreme_inputs
        )

        return configs

    def generate_and_save(
        self,
        reference_callable: Callable,
        shapes: dict,
        output_dir: str,
        model_config: dict | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> str:
        """Generate test configs and save as .pt files.

        Args:
            reference_callable: The reference function.
            shapes: Shape specification dict.
            output_dir: Directory to save test files.
            model_config: GPTConfig as dict.
            device: Device for tensors.
            dtype: Default dtype.

        Returns:
            Path to the output directory.
        """
        configs = self.generate(
            reference_callable, shapes, model_config, device, dtype
        )

        os.makedirs(output_dir, exist_ok=True)
        for config_name, (inputs, outputs) in configs.items():
            # Move to CPU for serialization
            cpu_inputs = [t.cpu() for t in inputs]
            cpu_outputs = [t.cpu() for t in outputs]
            save_path = os.path.join(output_dir, f"{config_name}.pt")
            torch.save({"inputs": cpu_inputs, "expected_outputs": cpu_outputs}, save_path)

        return output_dir

    def _extract_input_shapes(self, shapes: dict) -> list[list[int]]:
        """Extract input shapes from the shapes spec."""
        result = []
        inputs = shapes.get("inputs", [])
        for entry in inputs:
            shape = entry.get("shape", [])
            if shape and all(isinstance(d, int) for d in shape):
                result.append(shape)
        # Deduplicate
        seen = set()
        unique = []
        for s in result:
            key = tuple(s)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique if unique else []

    def _make_inputs(
        self,
        shapes: list[list[int]],
        scale_factor: float,
        min_dim: int,
        device: str,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Create random normal inputs with scaled shapes."""
        inputs = []
        for shape in shapes:
            scaled = [max(min_dim, int(d * scale_factor)) for d in shape]
            t = torch.randn(scaled, device=device, dtype=dtype)
            inputs.append(t)
        return inputs

    def _make_zero_inputs(
        self, shapes: list[list[int]], device: str, dtype: torch.dtype
    ) -> list[torch.Tensor]:
        """Create all-zero inputs."""
        return [torch.zeros(shape, device=device, dtype=dtype) for shape in shapes]

    def _make_extreme_inputs(
        self, shapes: list[list[int]], device: str, dtype: torch.dtype
    ) -> list[torch.Tensor]:
        """Create inputs with extreme values (mix of very large and small)."""
        inputs = []
        for shape in shapes:
            t = torch.randn(shape, device=device, dtype=dtype)
            # Mix in extreme values
            numel = t.numel()
            flat = t.view(-1)
            # Set ~10% of elements to large positive
            n_extreme = max(1, numel // 10)
            flat[:n_extreme] = 100.0
            # Set ~10% to large negative
            flat[n_extreme : 2 * n_extreme] = -100.0
            # Set ~5% to very small
            n_small = max(1, numel // 20)
            flat[2 * n_extreme : 2 * n_extreme + n_small] = 1e-6
            inputs.append(t)
        return inputs

    def _run_reference(
        self, reference_callable: Callable, inputs: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Run reference callable and capture outputs."""
        with torch.no_grad():
            output = reference_callable(*inputs)

        if isinstance(output, torch.Tensor):
            outputs = [output.detach().clone()]
        elif isinstance(output, tuple):
            outputs = [o.detach().clone() for o in output if isinstance(o, torch.Tensor)]
        else:
            outputs = [output]

        return (inputs, outputs)
