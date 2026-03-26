"""
TensorShapeDocumenter: documents tensor shape contracts from profiler
data for each fuseable group. Outputs shape specifications used by
test generation and kernel correctness verification.
"""

import json
import os

from ..schemas import FuseableGroup, OperationProfile, save_json


class TensorShapeDocumenter:
    """Documents tensor shape contracts for fuseable operation groups."""

    def document(
        self,
        group: FuseableGroup,
        profiles: list[OperationProfile],
    ) -> dict:
        """Build a shape specification from profiler data and group metadata.

        Args:
            group: The FuseableGroup with tensor_shapes from detection.
            profiles: Full list of OperationProfile (used to find matching ops).

        Returns:
            Dict with shape specification including input/output shapes,
            dtypes, and constraints.
        """
        shapes_spec = {
            "group_id": group.group_id,
            "fusion_type": group.fusion_type,
            "ops": group.op_names,
            "inputs": [],
            "outputs": [],
            "constraints": [],
        }

        # Gather shapes from the group's stored tensor_shapes
        for key, val in group.tensor_shapes.items():
            if "input" in key:
                for shape in val:
                    entry = {"name": key, "shape": shape, "dtype": "bfloat16"}
                    if entry not in shapes_spec["inputs"]:
                        shapes_spec["inputs"].append(entry)
            elif "output" in key:
                for shape in val:
                    entry = {"name": key, "shape": shape, "dtype": "bfloat16"}
                    if entry not in shapes_spec["outputs"]:
                        shapes_spec["outputs"].append(entry)

        # Also scan matching profiles for additional shape info
        group_op_set = set(group.op_names)
        for prof in profiles:
            if prof.op_name in group_op_set:
                for shape in prof.input_shapes:
                    entry = {"name": prof.op_name + "_input", "shape": shape, "dtype": "bfloat16"}
                    if entry not in shapes_spec["inputs"]:
                        shapes_spec["inputs"].append(entry)
                for shape in prof.output_shapes:
                    entry = {"name": prof.op_name + "_output", "shape": shape, "dtype": "bfloat16"}
                    if entry not in shapes_spec["outputs"]:
                        shapes_spec["outputs"].append(entry)

        # Infer constraints from shape patterns
        shapes_spec["constraints"] = self._infer_constraints(shapes_spec)

        return shapes_spec

    def document_and_save(
        self,
        group: FuseableGroup,
        profiles: list[OperationProfile],
        output_dir: str,
    ) -> str:
        """Document shapes and save to JSON.

        Args:
            group: The FuseableGroup.
            profiles: Full list of OperationProfile.
            output_dir: Base directory for reference_catalog.

        Returns:
            Path to the saved shapes.json file.
        """
        spec = self.document(group, profiles)
        group_dir = os.path.join(output_dir, group.group_id)
        os.makedirs(group_dir, exist_ok=True)
        path = os.path.join(group_dir, "shapes.json")
        save_json(spec, path)
        return path

    def _infer_constraints(self, shapes_spec: dict) -> list[str]:
        """Infer shape constraints from the collected shapes."""
        constraints = []
        all_shapes = [e["shape"] for e in shapes_spec["inputs"]] + [
            e["shape"] for e in shapes_spec["outputs"]
        ]

        if not all_shapes:
            return constraints

        # Check if all shapes have same rank
        ranks = {len(s) for s in all_shapes if s}
        if len(ranks) == 1:
            constraints.append(f"all_tensors_rank_{ranks.pop()}")

        # Check if last dimension is consistent
        last_dims = {s[-1] for s in all_shapes if s}
        if len(last_dims) == 1:
            constraints.append(f"uniform_last_dim_{last_dims.pop()}")

        # Check batch dimension consistency
        batch_dims = {s[0] for s in all_shapes if len(s) >= 2}
        if len(batch_dims) == 1:
            constraints.append(f"uniform_batch_dim_{batch_dims.pop()}")

        return constraints
