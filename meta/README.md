# Meta-Autoresearch

Optimizes the four-system optimization loop itself — bounded meta-optimization
of 30+ configurable harness parameters using a meta-level Thompson Sampling
bandit with fixed external evaluation (val_bpb on real data).

## Safety Argument

Based on the entropy decay proofs (arXiv:2601.05280), the meta-loop enforces:

1. **Fixed evaluator**: val_bpb on real validation data is NEVER modified
2. **Bounded search space**: only 30+ pre-defined harness parameters
3. **Shallow recursion**: exactly 1 level (meta-loop → inner loop)
4. **External grounding**: evaluation uses real data, not synthetic proxies

The meta-loop modifies *harness configuration only* — never the training code,
evaluation metric, dataset, or its own code.

## Architecture

```
Meta-Loop (meta_state.json)
├── Meta-Bandit: per-dimension Thompson Sampling over parameter variants
├── Meta-Experiments: K inner-loop iterations with experimental config
├── STOP Scaffold: LLM-generated harness strategy code snippets
├── Safety Sandbox: whitelist writes, recursion guard, budget enforcer
└── Knowledge Base: transferable insights across campaigns
```

## Key Parameters

| Parameter | Default | Range | System | Sensitivity |
|-----------|---------|-------|--------|-------------|
| T_base | 0.025 | [0.001, 0.5] | bandit | high |
| exploration_floor | 0.05 | [0.01, 0.3] | bandit | high |
| paper_preference_ratio | 0.4 | [0.0, 1.0] | bandit | medium |
| K_reheat_threshold | 5 | [2, 20] | bandit | medium |
| surrogate_retrain_threshold | 20 | [5, 50] | surrogate | medium |
| paper_filter_threshold | 0.3 | [0.1, 0.8] | surrogate | low |
| kernel_evolution_max_gens | 10 | [3, 30] | gpu_kernels | low |

See `meta_config_schema.json` for the full 30+ parameter inventory.

## Dashboard Interpretation

- **Meta-Regime**: baseline (collecting data), active (exploring), maintenance (converged)
- **Budget**: fraction of compute allocated to meta-experiments (default 20%)
- **ROI**: improvement attributable to meta-tuning / cost of meta-experiments
- **Dimension Posteriors**: per-parameter posterior mean shows confidence in variant selection
- **Convergence**: no promotions + low posterior variance = converged

## Tuning Guide

- If no parameters are being promoted: increase experiment length (more power)
- If too many experiments fail: reduce dimensions being explored
- If ROI is negative: reduce budget_fraction or enter maintenance mode
- If convergence detected prematurely: lower convergence_window threshold

## Troubleshooting

- **State corruption**: `MetaStateManager.recover()` replays from `meta_log.jsonl`
- **Safety boundary triggered**: investigate — never disable safety guards
- **Non-convergence**: check if landscape is non-stationary (DivergenceWatcher)
- **Budget overrun**: `MetaComputeBudgetEnforcer` auto-throttles

## Starting from Knowledge Base

```python
from meta.knowledge.bootstrapper import NewCampaignBootstrapper
bootstrapper = NewCampaignBootstrapper()
config = bootstrapper.bootstrap("meta_knowledge_base.json", "./new_campaign/")
```

## STOP Scaffold

Custom hooks are validated via AST analysis (`StrategySafetyChecker`).
Approved imports: `math`, `random`. No `os`, `subprocess`, `open()`, `exec()`.
Hook types: `selection_hook`, `acceptance_hook`, `prompt_hook`, `scheduling_hook`.
