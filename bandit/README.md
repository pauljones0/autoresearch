# Adaptive Bandit with Simulated Annealing -- Operator Guide

## 1. What the Bandit Does

The bandit replaces the fixed-fraction allocation strategy used by earlier
versions of autoresearch.  Instead of cycling through modification categories
in a predetermined order, it uses **Thompson Sampling** to learn which
categories are most productive and allocates iterations accordingly.

Each category is an "arm" in a multi-armed bandit.  The bandit maintains a
**Beta posterior** for every arm that tracks its observed success rate.  On
each iteration it samples from these posteriors and selects the arm with the
highest sample -- automatically balancing exploration of uncertain arms against
exploitation of known-good arms.

A **per-arm simulated annealing** layer sits on top of Thompson Sampling.
When an iteration produces a regression (negative delta), the annealing
decision may still accept it with probability exp(-|delta| / T), where T is
the arm's temperature.  Temperatures are derived from each arm's posterior
uncertainty and decay as evidence accumulates.  This lets the system escape
local optima early on and become increasingly greedy as confidence grows.

## 2. Arm Definitions

There are **9 canonical arms**:

| Arm ID | Display Name | Source Type | Dispatch Target |
|---|---|---|---|
| `architecture` | Architecture | internal | model_scientist architecture |
| `optimizer` | Optimizer | internal | model_scientist optimizer |
| `hyperparameter` | Hyperparameter | internal | model_scientist hyperparameter |
| `activation` | Activation | internal | model_scientist activation |
| `initialization` | Initialization | internal | model_scientist initialization |
| `regularization` | Regularization | internal | model_scientist regularization |
| `scheduling` | Scheduling | internal | model_scientist scheduling |
| `kernel_discovery` | Kernel Discovery | kernel | gpu_kernel discovery |
| `kernel_evolution` | Kernel Evolution | kernel | gpu_kernel evolution |

The 7 internal arms can have **paper variants** -- when a matching paper is
available in the Surrogate Triage queue and a random draw falls below
`paper_preference_ratio`, the arm dispatches through the paper-informed prompt
path instead of the internal prompt.

## 3. Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `T_base` | 0.025 | Base temperature scaling factor.  Higher = more annealing acceptance. |
| `K_reheat_threshold` | 5 | Consecutive failures before a reheat is triggered. |
| `exploration_floor` | 0.05 | Minimum per-arm selection probability (enforces exploration). |
| `paper_preference_ratio` | 0.4 | Probability of using a paper variant when available. |
| `reheat_factor` | 3.0 | Multiplier applied to temperature on reheat events. |
| `min_temperature` | 0.001 | Floor on per-arm temperature. |

Temperature derivation formula:

```
sigma_i = sqrt(alpha_i * beta_i / ((alpha_i + beta_i)^2 * (alpha_i + beta_i + 1)))
T_arm_i = max(min_temperature, T_base * sigma_i)
```

## 4. Dashboard Interpretation Guide

### Selection Entropy
Measures how evenly the bandit distributes selections.  Values near
`ln(n_arms)` (approx 2.2 for 9 arms) indicate uniform exploration.  Values
approaching 0 indicate the bandit has converged on one or two arms.

### Annealing Stepping-Stone Rate
Fraction of annealing-accepted (non-greedy) iterations that led to a
subsequent improvement.  A healthy range is 0.15--0.40.  Below 0.10 suggests
the temperature is too high (accepting too much junk).  Above 0.50 is rare
and typically means early-stage exploration is paying off.

### Posterior KL Divergence from Prior
Measures how far each arm's posterior has moved from the uninformative
Beta(1,1) prior.  Rising KL indicates the bandit is learning.  Stagnant KL
across many iterations may indicate the rewards are too noisy.

### Temperature Dispersion Ratio
Ratio of max to min temperature across arms.  A ratio above 10 indicates
some arms are still highly uncertain while others are well-characterised.
A ratio near 1 means all arms have similar exploration pressure.

### Regime Change Frequency
Number of detected regime changes (success rate drops) in a sliding window.
Zero is expected in steady state.  Persistent non-zero values indicate the
problem landscape is shifting.

## 5. Tuning Guide

**If the bandit is not exploring enough** (converges too fast on one arm):
- Increase `exploration_floor` (e.g., 0.05 -> 0.10).
- Increase `T_base` to accept more regressions.

**If the bandit wastes too many iterations on bad arms**:
- Decrease `T_base` (e.g., 0.025 -> 0.015).
- Lower `exploration_floor`.
- Increase `K_reheat_threshold` to be less aggressive about reheating.

**If reheats are too frequent**:
- Increase `K_reheat_threshold` (e.g., 5 -> 8).
- Lower `reheat_factor` (e.g., 3.0 -> 2.0).

**If paper-based iterations are underperforming**:
- Lower `paper_preference_ratio`.
- Check the Surrogate Triage queue quality.

## 6. Troubleshooting

### State Corruption Recovery
If `strategy_state.json` is corrupted or inconsistent:
1. Run the health auditor: `BanditHealthAuditor().audit(...)`.
2. Run the replay validator: `LogReplayValidator().validate(state_path, log_path)`.
3. If replay succeeds, replace the saved state with the replayed state.
4. If replay also fails, re-initialise from the warm-start entries in the
   journal using the `StateManager.warm_start()` path.

### Regime Downgrade
If the bandit is in `full_bandit` but performance degrades:
1. Check `regime_change_frequency` -- if elevated, the landscape shifted.
2. Consider resetting posteriors for affected arms (see below).
3. As a last resort, set `regime` to `conservative_bandit` which doubles
   exploration floors.

### Adding / Removing Arms
1. Add the arm definition to `bandit/taxonomy.py`.
2. The warm-start process will initialise it with Beta(1,1).
3. Existing arm posteriors are unaffected.
4. To remove an arm, delete it from the taxonomy and the state.  Evidence
   conservation checks will flag the discrepancy -- re-run warm-start.

### Resetting Posteriors
To reset a single arm to the prior:
```python
state.arms["arm_id"].alpha = 1.0
state.arms["arm_id"].beta = 1.0
state.arms["arm_id"].consecutive_failures = 0
state.arms["arm_id"].reheat_count = 0
```
Then recalculate the temperature and save the state.

## 7. Cross-System Interactions

- **Model Scientist Pipeline**: Internal arms dispatch modifications through
  this pipeline.  The bandit reads the returned delta (score change) to update
  posteriors.
- **Surrogate Triage Pipeline**: Paper variants pull entries from the
  Surrogate Triage queue.  The `PaperArmSplitter` decides routing.
- **GPU Kernel Pipeline**: The two kernel arms dispatch through kernel
  discovery and evolution pipelines respectively.
- **Journal**: The bandit reads journal entries during warm-start and writes
  `bandit_arm` annotations to new entries.
- **Diagnostics**: The `diagnostics_boost` field on each arm can be set by
  external diagnostics to temporarily favour an arm.

## 8. Mathematical Appendix

### Beta Distribution
Each arm's success rate is modelled as a Beta(alpha, beta) distribution.
- Mean: alpha / (alpha + beta)
- Variance: alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))
- After a success: alpha += 1.  After a failure: beta += 1.
- Prior: Beta(1, 1) = Uniform(0, 1).

### Thompson Sampling
On each iteration, sample theta_i ~ Beta(alpha_i, beta_i) for every arm i.
Select the arm with the highest theta_i.  This naturally balances exploration
(arms with high variance get lucky draws) and exploitation (arms with high
mean tend to win).

### Simulated Annealing
When an iteration produces delta < 0:
```
T_effective = T_arm * (1 + constraint_density) * surrogate_modulation_factor
P(accept) = exp(delta / T_effective)
```
Accept with probability P.  As T decays toward min_temperature, acceptance of
regressions approaches zero and the system becomes greedy.

### Reheating
When an arm accumulates K_reheat_threshold consecutive failures, its
temperature is multiplied by reheat_factor.  This allows re-exploration of
arms that may have hit a temporary dead end.  Reheat budget is finite to
prevent infinite oscillation.

### KL Divergence from Prior
```
KL(Beta(a,b) || Beta(1,1)) = -ln B(a,b) + (a-1)*psi(a) + (b-1)*psi(b) - (a+b-2)*psi(a+b)
```
where psi is the digamma function and B is the Beta function.
