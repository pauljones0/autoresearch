"""
Literature-Informed Surrogate Triage Pipeline

Ingests techniques from arXiv, scores them with a surrogate model trained
on Model Scientist data, and routes the best candidates through the existing
diagnostic → scale-gate → ablation → journal pipeline.
"""

__version__ = "0.1.0"
