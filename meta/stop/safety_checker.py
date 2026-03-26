"""Static safety analysis for generated strategy code."""

import ast

from meta.schemas import GeneratedStrategy, SafetyCheckResult


# Modules that strategies must never import
_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "shutil", "pathlib", "socket", "http", "urllib",
})

# Built-in calls that strategies must never invoke
_FORBIDDEN_CALLS = frozenset({
    "open", "exec", "eval", "compile", "__import__",
})

_MAX_LINES = 200


class StrategySafetyChecker:
    """AST-based static safety checker for generated strategy code."""

    def check(self, strategy: GeneratedStrategy) -> SafetyCheckResult:
        """Check a strategy for safety violations.

        Returns a SafetyCheckResult with safe=True if no violations found.
        """
        violations: list = []
        nodes_checked = 0

        # Check line count
        lines = strategy.code.strip().splitlines()
        if len(lines) > _MAX_LINES:
            violations.append(
                f"Code exceeds {_MAX_LINES} lines (has {len(lines)})"
            )

        # Parse AST
        try:
            tree = ast.parse(strategy.code)
        except SyntaxError as exc:
            violations.append(f"SyntaxError: {exc}")
            return SafetyCheckResult(
                safe=False,
                violations=violations,
                ast_nodes_checked=0,
            )

        # Walk all AST nodes
        for node in ast.walk(tree):
            nodes_checked += 1

            # Check imports: import os / import subprocess ...
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split(".")[0]
                    if top_module in _FORBIDDEN_MODULES:
                        violations.append(
                            f"Forbidden import: {alias.name}"
                        )

            # Check from-imports: from os import path ...
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_module = node.module.split(".")[0]
                    if top_module in _FORBIDDEN_MODULES:
                        violations.append(
                            f"Forbidden from-import: {node.module}"
                        )

            # Check function calls: open(), exec(), eval() ...
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name in _FORBIDDEN_CALLS:
                    violations.append(
                        f"Forbidden call: {call_name}()"
                    )

        return SafetyCheckResult(
            safe=len(violations) == 0,
            violations=violations,
            ast_nodes_checked=nodes_checked,
        )

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        """Extract the function name from a Call node."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""
