"""
Meta-Autoresearch Pipeline

Optimizes the four-system optimization loop itself — bounded meta-optimization
of 30+ configurable harness parameters using a meta-level Thompson Sampling
bandit with fixed external evaluation (val_bpb on real data).

Safety: The meta-loop modifies harness configuration ONLY — never the
evaluation metric, dataset, training code, or its own code.
"""

__version__ = "0.1.0"
