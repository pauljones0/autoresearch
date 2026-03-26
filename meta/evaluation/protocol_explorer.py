"""
Evaluation protocol explorer — generate evaluation protocol variants.
"""

from meta.schemas import EvalProtocol


class EvalProtocolExplorer:
    """Generate evaluation protocol variants."""

    def generate_protocols(self) -> list:
        return [
            EvalProtocol(protocol_id="fast_cheap", training_steps=500,
                         n_seeds=1, warmup_fraction=0.1),
            EvalProtocol(protocol_id="standard", training_steps=1000,
                         n_seeds=1, warmup_fraction=0.2),
            EvalProtocol(protocol_id="careful", training_steps=1500,
                         n_seeds=2, warmup_fraction=0.2),
            EvalProtocol(protocol_id="thorough", training_steps=2000,
                         n_seeds=3, warmup_fraction=0.25),
            EvalProtocol(protocol_id="two_stage", training_steps=1500,
                         n_seeds=2, warmup_fraction=0.2, is_two_stage=True,
                         stage1_steps=500, stage1_seeds=1,
                         stage2_steps=1500, stage2_seeds=2),
        ]
