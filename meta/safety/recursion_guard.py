"""
Recursion depth guard — prevents meta-meta-optimization.
"""

import os
from meta.schemas import BoundaryViolationError


class RecursionDepthGuard:
    """Hard stop against recursive meta-loop launches."""

    ENV_VAR = "META_RECURSION_DEPTH"
    MAX_DEPTH = 1

    def check_depth(self) -> int:
        depth = int(os.environ.get(self.ENV_VAR, "0"))
        if depth >= self.MAX_DEPTH:
            raise BoundaryViolationError(
                "recursive_meta",
                f"Recursion depth {depth} >= max {self.MAX_DEPTH}",
            )
        return depth

    @staticmethod
    def set_depth(depth: int):
        os.environ["META_RECURSION_DEPTH"] = str(depth)
