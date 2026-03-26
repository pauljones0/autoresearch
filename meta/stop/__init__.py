"""STOP (Self-Taught Optimizer) scaffold for meta-strategy generation."""

from meta.stop.scaffold import STOPScaffold
from meta.stop.safety_checker import StrategySafetyChecker
from meta.stop.executor import StrategyExecutor
from meta.stop.evolution import StrategyEvolutionController

__all__ = [
    "STOPScaffold",
    "StrategySafetyChecker",
    "StrategyExecutor",
    "StrategyEvolutionController",
]
