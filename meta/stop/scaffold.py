"""STOP scaffold: generates strategy code snippets for harness hooks."""

import hashlib
import time
from typing import List

from meta.schemas import GeneratedStrategy, MetaExperimentResult


# ---------------------------------------------------------------------------
# Strategy templates for each hook type
# ---------------------------------------------------------------------------

_SELECTION_TEMPLATES: list = [
    {
        "description": "Epsilon-greedy selection with decaying epsilon",
        "code": (
            "import random\n"
            "import math\n"
            "\n"
            "def selection_hook(candidates, scores, iteration, **kwargs):\n"
            "    epsilon = max(0.05, 1.0 / (1.0 + 0.01 * iteration))\n"
            "    if random.random() < epsilon:\n"
            "        return random.choice(candidates)\n"
            "    best_idx = scores.index(max(scores))\n"
            "    return candidates[best_idx]\n"
        ),
        "rationale": "Decaying epsilon-greedy balances exploration early and exploitation later.",
    },
    {
        "description": "UCB1-style selection with confidence bonus",
        "code": (
            "import math\n"
            "\n"
            "def selection_hook(candidates, scores, iteration, counts=None, **kwargs):\n"
            "    if counts is None:\n"
            "        counts = [1] * len(candidates)\n"
            "    total = sum(counts)\n"
            "    ucb_scores = [\n"
            "        s + math.sqrt(2.0 * math.log(total + 1) / (c + 1))\n"
            "        for s, c in zip(scores, counts)\n"
            "    ]\n"
            "    best_idx = ucb_scores.index(max(ucb_scores))\n"
            "    return candidates[best_idx]\n"
        ),
        "rationale": "UCB1 provides theoretical exploration guarantees via confidence bounds.",
    },
    {
        "description": "Boltzmann (softmax) selection with temperature decay",
        "code": (
            "import math\n"
            "import random\n"
            "\n"
            "def selection_hook(candidates, scores, iteration, **kwargs):\n"
            "    temperature = max(0.1, 5.0 / (1.0 + 0.02 * iteration))\n"
            "    max_s = max(scores) if scores else 0\n"
            "    weights = [math.exp((s - max_s) / temperature) for s in scores]\n"
            "    total_w = sum(weights)\n"
            "    probs = [w / total_w for w in weights]\n"
            "    r = random.random()\n"
            "    cumulative = 0.0\n"
            "    for i, p in enumerate(probs):\n"
            "        cumulative += p\n"
            "        if r <= cumulative:\n"
            "            return candidates[i]\n"
            "    return candidates[-1]\n"
        ),
        "rationale": "Softmax selection allows smooth transition from exploration to exploitation.",
    },
]

_ACCEPTANCE_TEMPLATES: list = [
    {
        "description": "Simulated annealing acceptance with cooling schedule",
        "code": (
            "import math\n"
            "import random\n"
            "\n"
            "def acceptance_hook(delta, iteration, **kwargs):\n"
            "    if delta > 0:\n"
            "        return True\n"
            "    temperature = max(0.001, 1.0 * (0.995 ** iteration))\n"
            "    prob = math.exp(delta / temperature)\n"
            "    return random.random() < prob\n"
        ),
        "rationale": "SA acceptance allows uphill moves early, converging to greedy over time.",
    },
    {
        "description": "Threshold acceptance with adaptive threshold",
        "code": (
            "def acceptance_hook(delta, iteration, **kwargs):\n"
            "    threshold = max(-0.01, -0.5 / (1.0 + 0.05 * iteration))\n"
            "    return delta > threshold\n"
        ),
        "rationale": "Threshold acceptance provides deterministic control with gradual tightening.",
    },
]

_PROMPT_TEMPLATES: list = [
    {
        "description": "Structured chain-of-thought prompt augmentation",
        "code": (
            "def prompt_hook(base_prompt, context, **kwargs):\n"
            "    augmentation = (\n"
            "        '\\nApproach this step-by-step:\\n'\n"
            "        '1. Identify the core bottleneck in the current code.\\n'\n"
            "        '2. Propose a targeted fix that addresses only this bottleneck.\\n'\n"
            "        '3. Verify the fix does not regress other metrics.\\n'\n"
            "    )\n"
            "    return base_prompt + augmentation\n"
        ),
        "rationale": "Chain-of-thought prompting improves reasoning quality.",
    },
    {
        "description": "Priority-weighted context injection",
        "code": (
            "def prompt_hook(base_prompt, context, **kwargs):\n"
            "    history = context.get('recent_deltas', [])\n"
            "    if history:\n"
            "        recent_avg = sum(history[-5:]) / min(len(history), 5)\n"
            "        if recent_avg < 0:\n"
            "            extra = '\\nRecent attempts have been declining. Focus on stability.\\n'\n"
            "        else:\n"
            "            extra = '\\nRecent progress is positive. Try bolder changes.\\n'\n"
            "        return base_prompt + extra\n"
            "    return base_prompt\n"
        ),
        "rationale": "Adaptive context based on recent performance steers LLM focus.",
    },
]

_SCHEDULING_TEMPLATES: list = [
    {
        "description": "Round-robin dimension scheduling with priority boost",
        "code": (
            "def scheduling_hook(dimensions, scores, iteration, **kwargs):\n"
            "    n = len(dimensions)\n"
            "    if n == 0:\n"
            "        return None\n"
            "    base_idx = iteration % n\n"
            "    if scores and max(scores) - min(scores) > 0.1:\n"
            "        best_idx = scores.index(max(scores))\n"
            "        if iteration % 3 == 0:\n"
            "            return dimensions[best_idx]\n"
            "    return dimensions[base_idx]\n"
        ),
        "rationale": "Round-robin ensures coverage; priority boost exploits promising dimensions.",
    },
    {
        "description": "Variance-weighted dimension scheduling",
        "code": (
            "import random\n"
            "\n"
            "def scheduling_hook(dimensions, scores, iteration, variances=None, **kwargs):\n"
            "    if not dimensions:\n"
            "        return None\n"
            "    if variances and len(variances) == len(dimensions):\n"
            "        total_var = sum(variances)\n"
            "        if total_var > 0:\n"
            "            probs = [v / total_var for v in variances]\n"
            "            r = random.random()\n"
            "            cumulative = 0.0\n"
            "            for i, p in enumerate(probs):\n"
            "                cumulative += p\n"
            "                if r <= cumulative:\n"
            "                    return dimensions[i]\n"
            "    return dimensions[iteration % len(dimensions)]\n"
        ),
        "rationale": "High-variance dimensions get more exploration budget.",
    },
]

_TEMPLATES = {
    "selection_hook": _SELECTION_TEMPLATES,
    "acceptance_hook": _ACCEPTANCE_TEMPLATES,
    "prompt_hook": _PROMPT_TEMPLATES,
    "scheduling_hook": _SCHEDULING_TEMPLATES,
}


class STOPScaffold:
    """Generates template-based strategy code for harness hooks."""

    HOOK_TYPES = ("selection_hook", "acceptance_hook", "prompt_hook", "scheduling_hook")

    def generate_strategy(
        self,
        current_best_config: dict,
        experiment_history: List[MetaExperimentResult],
        baseline_ir: float,
    ) -> GeneratedStrategy:
        """Generate a strategy for the most promising hook type.

        Picks the hook type that has been least explored in experiment_history,
        then selects a template that hasn't been tried yet (cycling through).
        """
        hook_type = self._pick_hook_type(experiment_history)
        template = self._pick_template(hook_type, experiment_history)

        strategy_id = self._make_id(hook_type, template["code"])

        return GeneratedStrategy(
            strategy_id=strategy_id,
            hook_type=hook_type,
            code=template["code"],
            description=template["description"],
            llm_rationale=template["rationale"],
            estimated_improvement=f"baseline_ir={baseline_ir if baseline_ir is not None else 0.0:.4f}",
        )

    def generate_for_hook(
        self,
        hook_type: str,
        index: int = 0,
    ) -> GeneratedStrategy:
        """Generate a specific strategy by hook type and template index."""
        if hook_type not in _TEMPLATES:
            raise ValueError(f"Unknown hook_type: {hook_type}")
        templates = _TEMPLATES[hook_type]
        template = templates[index % len(templates)]
        strategy_id = self._make_id(hook_type, template["code"])
        return GeneratedStrategy(
            strategy_id=strategy_id,
            hook_type=hook_type,
            code=template["code"],
            description=template["description"],
            llm_rationale=template["rationale"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_hook_type(self, history: List[MetaExperimentResult]) -> str:
        """Pick the least-explored hook type."""
        counts = {h: 0 for h in self.HOOK_TYPES}
        for exp in history:
            for diff in exp.config_diff:
                param_id = diff.get("param_id", "") if isinstance(diff, dict) else getattr(diff, "param_id", "")
                for hook in self.HOOK_TYPES:
                    if hook in param_id:
                        counts[hook] += 1
        return min(counts, key=counts.get)

    def _pick_template(self, hook_type: str, history: List[MetaExperimentResult]) -> dict:
        """Pick the next template in rotation for a hook type."""
        templates = _TEMPLATES[hook_type]
        n_prior = sum(
            1 for exp in history
            for diff in exp.config_diff
            if hook_type in (
                diff.get("param_id", "") if isinstance(diff, dict)
                else getattr(diff, "param_id", "")
            )
        )
        return templates[n_prior % len(templates)]

    @staticmethod
    def _make_id(hook_type: str, code: str) -> str:
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        return f"stop_{hook_type}_{code_hash}"
