# Meta-Autoresearch — Multi-Phase Implementation Plan

**Source:** Synergy E from `ideas_v2.md` (Risk 4/7, Reward 5/7)
**Status:** NOT IMPLEMENTED
**Depends on:**
- Model Scientist Pipeline (Synergy C) — **IMPLEMENTED** (`model_scientist/`)
- Surrogate Triage Pipeline (Synergy B) — **IMPLEMENTED** (`surrogate_triage/`)
- GPU Kernel Creation Pipeline (Synergy D) — **IMPLEMENTED** (`gpu_kernels/`)
- Adaptive Bandit with Simulated Annealing (Synergy A) — **IMPLEMENTED** (`bandit/`)

**Goal:** Optimize the four-system optimization loop itself — bounded meta-optimization of the 30+ configurable harness parameters across all four systems (bandit arm definitions, annealing temperatures, surrogate retraining triggers, kernel discovery schedules, prompt templates, context allocation, evaluation protocol) using a meta-level Thompson Sampling bandit with fixed external evaluation (val_bpb on real data). The meta-loop treats the entire four-system inner loop as a black box, proposes harness configuration changes, runs K inner-loop iterations under each configuration, scores by improvement rate, and promotes winning configurations — while respecting the mathematical limits on recursive self-improvement (arXiv:2601.05280): fixed evaluator, bounded search space, shallow recursion (exactly 1 level), external grounding.

**Critical Safety Constraint:** The meta-loop modifies *harness configuration only* — it NEVER modifies the evaluation metric (val_bpb on real validation data), the dataset, the training code (`train.py`), or the meta-loop's own code. This is not optional; the entropy decay proofs (arXiv:2601.05280) show that self-modifying evaluation leads to provable degradation. The meta-loop is a parameter tuner for the research harness, not a recursive self-improver.

---

## Dependency Map: What the Four Implemented Systems Provide

### From Adaptive Bandit (`bandit/`)

| Component | File | How Meta-Loop Uses It |
|---|---|---|
| `BanditState` schema | `bandit/schemas.py` | The meta-loop's primary optimization target — `T_base`, `exploration_floor`, `paper_preference_ratio`, `K_reheat_threshold`, `reheat_factor`, `min_temperature`, and per-arm `diagnostics_boost` weights are all meta-parameters that the meta-loop can modify via `bandit_overrides.json` |
| `HotConfigReloader` | `bandit/config_reload.py` | The mechanism by which the meta-loop injects configuration changes — it writes to `bandit_overrides.json` and the inner loop picks up changes on the next iteration without restart. The meta-loop never directly modifies `strategy_state.json` |
| `AdaptiveBanditPipeline.run_iteration()` | `bandit/pipeline.py` | The inner loop's single-iteration entry point — the meta-loop calls this K times to evaluate a configuration, then reads the results from `hypothesis_journal.jsonl` and `bandit_log.jsonl` |
| `BanditMetricComputer.compute_all()` | `bandit/metrics.py` | Provides the 5 bandit-derived metrics (`arm_selection_entropy`, `annealing_stepping_stone_rate`, `posterior_kl_divergence_from_prior`, `temperature_dispersion_ratio`, `regime_change_frequency`) as features for the meta-bandit's contextual decisions |
| `AutoTuner.recommend()` | `bandit/tuning.py` | The meta-loop subsumes and extends the inner-loop `AutoTuner` — when the meta-loop is active, the inner-loop `AutoTuner` is disabled (`enable_auto_tuning: False` in `strategy_state.json`) to prevent conflicting parameter changes. The meta-loop's recommendations are strictly superior because they're based on multi-iteration evaluation windows, not single-iteration heuristics |
| `ABTestOrchestrator` | `bandit/ab_test.py` | Provides the A/B testing infrastructure that the meta-loop reuses — each meta-experiment is essentially a controlled A/B test of an experimental configuration vs. the current best |
| `BanditLogWriter` | `bandit/log.py` | The meta-loop reads `bandit_log.jsonl` to extract per-iteration metrics (which arm was selected, what temperature was used, whether annealing was triggered) for computing improvement rate under each configuration |
| `AcceptanceAnalyzer` | `bandit/analysis/acceptance_analyzer.py` | Provides `stepping_stone_rate` and `dead_end_rate` which are secondary meta-evaluation metrics — a configuration that produces higher stepping-stone rate is more valuable even if its raw improvement rate is similar |
| `CategoryPromptRouter` | `bandit/prompt_router.py` | The prompt templates in `bandit/prompt_templates/` are meta-parameters — the meta-loop generates prompt variants and writes them to these files, and the inner loop picks them up on the next iteration |
| `PosteriorVisualizer` | `bandit/visualization.py` | The meta-loop displays meta-bandit posteriors alongside inner-bandit posteriors in a nested dashboard |

### From Model Scientist (`model_scientist/`)

| Component | File | How Meta-Loop Uses It |
|---|---|---|
| `ModelScientistPipeline` | `pipeline.py` | Provides configurable parameters that become meta-dimensions: `scale_gate_enabled` (bool), `ablation_enabled` (bool), `min_training_steps` (int), `max_training_steps` (int), `n_eval_seeds` (int), `diagnostics_capture_interval` (int), `failure_mining_interval` (int) |
| `DiagnosticsInstrumenter` | `diagnostics/instrumenter.py` | The `capture_interval` (how often diagnostics are collected) is a meta-parameter — more frequent capture provides richer signals to the inner bandit but costs compute |
| `MetricRegistry` | `metrics/registry.py` | The metric evolution cycle's parameters (`correlation_threshold_for_pruning`, `max_active_metrics`, `proposal_frequency`) are meta-parameters — the meta-loop can tune how aggressively the system prunes or adopts new metrics |
| `SafetyGuard` | `integration/safety_guard.py` | The safety guard's compute budget is a meta-parameter — the meta-loop allocates total compute between inner-loop experiments and meta-experiments (default 80/20 split) |
| `PipelineMonitor` | `integration/monitor.py` | The meta-loop adds a "Meta-Optimization" section to the unified dashboard showing meta-bandit posteriors, meta-experiment history, and the current exploration/exploitation split |
| `JournalWriter` / `JournalReader` | `journal/writer.py`, `reader.py` | The journal is the primary data source for computing improvement rate — the meta-loop reads accepted deltas from the journal to score configurations |
| `CriticAgent` | `metrics/critic.py` | The critic's proposal frequency is a meta-parameter — running the critic every iteration costs context but may improve diagnostic quality faster |
| `FailureClusterer` | `failure_mining/clusterer.py` | The clustering parameters (`min_cluster_size`, `merge_threshold`) are meta-parameters — tighter clustering produces more constraints, which may help or hurt the inner bandit |

### From Surrogate Triage (`surrogate_triage/`)

| Component | File | How Meta-Loop Uses It |
|---|---|---|
| `SurrogateTriagePipeline` | `pipeline.py` | Provides configurable parameters: `paper_evaluation_fraction` (now controlled by bandit but default fallback), `daily_ingestion_days_back`, `daily_ingestion_max_results`, `top_n_candidates_to_queue` |
| `SurrogateRetrainer` | `surrogate/retrainer.py` | The retraining trigger `min_new_entries` (default 20) is a meta-parameter — the meta-loop can test whether more or less frequent retraining improves overall improvement rate |
| `SurrogateTrainer` | `surrogate/trainer.py` | Training hyperparameters (`epochs`, `lr`, `k_folds`, `patience`) are meta-parameters — the meta-loop can tune the surrogate's own training to improve prediction quality |
| `ColdStartManager` | `surrogate/cold_start.py` | The regime thresholds (50 for conservative, 200 for full) are meta-parameters — the meta-loop can test whether earlier or later surrogate engagement improves improvement rate |
| `QueueManager` | `funnel/queue_manager.py` | `max_queue_size` (default 50) and `dedup_threshold` (default 0.95) are meta-parameters |
| `PaperFilter` | `ingestion/paper_filter.py` | The `threshold` (default 0.3) for paper relevance is a meta-parameter — lower threshold = more papers = more diverse candidates but more noise |
| `EvaluationScheduler` | `funnel/evaluation_scheduler.py` | The fixed fractions (internal 60%, paper 25%, kernel 15%) are the pre-bandit defaults — the meta-loop can tune these fallback values for the `no_bandit` regime |
| `KnowledgeCeilingMonitor` | `steady_state/ceiling_monitor.py` | The `window_size` (default 50) and trend thresholds are meta-parameters — the meta-loop can test whether different window sizes produce more actionable ceiling signals |

### From GPU Kernels (`gpu_kernels/`)

| Component | File | How Meta-Loop Uses It |
|---|---|---|
| `GPUKernelPipeline` | `pipeline.py` | Provides configurable parameters: discovery trigger frequency, verification thoroughness (n_runs in stability), benchmark iterations |
| `EvolutionaryRefinementScheduler` | `evolution/scheduler.py` | `recency_cooldown_seconds` (default 3600), min bandwidth for refinement candidate (0.6) are meta-parameters |
| `KernelBenchmarker` | `benchmarking/benchmarker.py` | `n_warmup` (50), `n_timed` (200), `n_runs` (5) are meta-parameters — cheaper benchmarking saves compute but increases variance |
| `EvolutionConvergenceDetector` | `evolution/convergence.py` | `max_generations` (10), min improvement threshold (1%) are meta-parameters |
| `RuntimeCorrectnessMonitor` | `monitoring/runtime_monitor.py` | `check_interval` (100) is a meta-parameter — more frequent checks catch errors earlier but cost compute |
| `KernelMutationEngine` | `evolution/mutator.py` | `n_mutations` (5) per generation is a meta-parameter |

---

## Phase 1: Meta-Parameter Inventory, Externalization & Baseline Measurement

**Gate:** Phase 1 is COMPLETE when: (1) `meta_config_schema.json` enumerates at least 30 independently adjustable harness parameters across all four systems, each with: current default value, valid range (with JSON Schema type/min/max), system of origin (`bandit`, `model_scientist`, `surrogate_triage`, `gpu_kernels`), expected impact hypothesis, and measurement sensitivity estimate; (2) all 30+ parameters are readable from a unified `meta_config.json` file and each system reads its parameters from this file at runtime (not hardcoded); (3) the meta-safety sandbox passes all 5 boundary violation tests (see Subphase 1.3); (4) 3 independent 100-iteration baseline runs establish the improvement rate distribution with measured mean and variance; (5) the minimum detectable effect size is computed (< 20% relative improvement in improvement rate) given the measured variance.

### Subphase 1.1: Four-System Meta-Parameter Inventory

**Status:** NOT IMPLEMENTED
**Gate:** Passes when `meta_config_schema.json` contains at least 30 parameters with complete metadata, validated against all four systems' actual code — every parameter has been traced to the exact line of code that reads it, confirming it's genuinely adjustable without side effects.

| Agent | Task |
|-------|------|
| **BanditParameterInventorist** | Build `meta/inventory/bandit_params.py`. Class `BanditParameterInventorist` with method `inventory() → list[MetaParameter]`. Enumerate all adjustable parameters in `bandit/`: (1) `T_base` (float, [0.001, 0.5], default 0.025, from `BanditState`), (2) `exploration_floor` (float, [0.01, 0.3], default 0.05), (3) `paper_preference_ratio` (float, [0.0, 1.0], default 0.4), (4) `K_reheat_threshold` (int, [2, 20], default 5), (5) `reheat_factor` (float, [1.0, 10.0], default 3.0), (6) `min_temperature` (float, [0.0001, 0.01], default 0.001), (7) `enable_rollback_safety` (bool, default True), (8) each `BOOST_RULES` weight in `bandit/boosting.py` (e.g., `attention_entropy_high_architecture_boost`, default 1.5 — 9 rule weights total). For each parameter: trace the exact code path that reads it (e.g., `T_base` is read in `bandit/temperature.py:TemperatureDeriver.compute()` and `bandit/updater.py:PosteriorUpdateEngine._compute_temperature()`), verify that changing it has no side effects beyond the intended behavior (e.g., changing `T_base` doesn't corrupt state), and document the impact hypothesis ("Increasing T_base makes the bandit more tolerant of regressions, which may help escape local optima but risks accepting too many bad modifications"). `MetaParameter` dataclass: `{param_id: str, display_name: str, system: str, type: str ("float"|"int"|"bool"|"str"), default_value, valid_range: dict (min/max or enum), current_value, code_path: str (file:class.method), impact_hypothesis: str, sensitivity_estimate: str ("high"|"medium"|"low"|"unknown"), category: str ("temperature"|"exploration"|"evaluation"|"scheduling"|"prompt"|"budget")}`. |
| **ModelScientistParameterInventorist** | Build `meta/inventory/ms_params.py`. Class `ModelScientistParameterInventorist` with method `inventory() → list[MetaParameter]`. Enumerate: (1) `scale_gate_enabled` (bool, default True, from `ModelScientistPipeline.__init__`), (2) `ablation_enabled` (bool, default True), (3) `min_training_steps` (int, [100, 5000], from the training evaluation configuration), (4) `max_training_steps` (int, [500, 20000]), (5) `n_eval_seeds` (int, [1, 5], default 1), (6) `diagnostics_capture_interval` (int, [1, 50], default 10 — how many iterations between full diagnostic captures), (7) `failure_mining_interval` (int, [5, 100], default 20), (8) `metric_correlation_threshold` (float, [0.1, 0.9], default 0.3 — threshold in `MetricRegistry` for pruning uncorrelated metrics), (9) `max_active_metrics` (int, [5, 30], default 15), (10) `critic_proposal_frequency` (int, [1, 20], default 5 — iterations between `CriticAgent` proposals). For each: trace the code path, verify adjustability, document hypothesis. |
| **SurrogateTriageParameterInventorist** | Build `meta/inventory/st_params.py`. Class `SurrogateTriageParameterInventorist` with method `inventory() → list[MetaParameter]`. Enumerate: (1) `surrogate_retrain_threshold` (int, [5, 50], default 20 from `SurrogateRetrainer.min_new_entries`), (2) `surrogate_training_epochs` (int, [20, 500], default 100), (3) `surrogate_learning_rate` (float, [1e-4, 1e-2], default 1e-3), (4) `surrogate_k_folds` (int, [2, 10], default 5), (5) `cold_start_conservative_threshold` (int, [20, 100], default 50), (6) `cold_start_full_threshold` (int, [100, 500], default 200), (7) `max_queue_size` (int, [10, 200], default 50), (8) `dedup_threshold` (float, [0.7, 1.0], default 0.95), (9) `paper_filter_threshold` (float, [0.1, 0.8], default 0.3), (10) `daily_ingestion_max_results` (int, [10, 500], default 100), (11) `ceiling_monitor_window_size` (int, [20, 200], default 50). |
| **GPUKernelParameterInventorist** | Build `meta/inventory/gk_params.py`. Class `GPUKernelParameterInventorist` with method `inventory() → list[MetaParameter]`. Enumerate: (1) `kernel_benchmark_n_warmup` (int, [10, 200], default 50), (2) `kernel_benchmark_n_timed` (int, [50, 1000], default 200), (3) `kernel_benchmark_n_runs` (int, [1, 10], default 5), (4) `kernel_evolution_max_generations` (int, [3, 30], default 10), (5) `kernel_evolution_min_improvement` (float, [0.001, 0.1], default 0.01), (6) `kernel_refinement_cooldown` (int, [600, 7200], default 3600), (7) `kernel_runtime_check_interval` (int, [10, 500], default 100), (8) `kernel_mutations_per_generation` (int, [2, 20], default 5), (9) `kernel_discovery_block_sizes` (list[int], default [128, 256, 512]). |
| **MetaConfigSchemaBuilder** | Build `meta/config_schema.py`. Class `MetaConfigSchemaBuilder` with method `build(inventories: list[list[MetaParameter]]) → dict`. Aggregates all parameters from the 4 inventorists into a single `meta_config_schema.json` following JSON Schema draft-07 format. Each parameter becomes a property with type, minimum, maximum, default, description, and custom `x-system`, `x-category`, `x-code-path`, `x-hypothesis` fields. The schema is used by `MetaConfigManager` to validate any proposed configuration before applying it. Also generates `meta_config.json` with all current default values. |

### Subphase 1.2: Unified Configuration Externalization

**Status:** NOT IMPLEMENTED
**Gate:** Passes when: (1) `meta_config.json` exists with all 30+ parameters; (2) each of the 4 systems reads its parameters from this file at runtime — verified by: modifying a parameter in `meta_config.json`, running 1 iteration, and confirming the system used the new value (logged in the relevant system's log); (3) modifying `meta_config.json` does NOT require restarting any system; (4) if `meta_config.json` is missing or corrupted, all 4 systems fall back to their hardcoded defaults (graceful degradation).

| Agent | Task |
|-------|------|
| **MetaConfigManager** | Build `meta/config_manager.py`. Class `MetaConfigManager` with methods: `load(config_path="meta_config.json") → dict` — load and validate against `meta_config_schema.json`, returning the merged configuration (file values override defaults); `save(config: dict, config_path)` — atomic write (temp + rename, same pattern as `AtomicStateManager`); `get_param(param_id) → any` — look up a single parameter; `set_param(param_id, value) → bool` — set a single parameter with validation; `get_system_params(system: str) → dict` — return all parameters for one system; `diff(config_a, config_b) → list[ParamDiff]` — show what changed between two configurations; `validate(config) → list[str]` — return list of validation errors. The manager caches the config with an mtime check — if the file has been modified since last load, it reloads. This enables the meta-loop to write changes and have all four systems pick them up within one iteration. |
| **BanditConfigBridge** | Build `meta/bridges/bandit_bridge.py`. Class `BanditConfigBridge` with method `apply(meta_config: dict)`. Writes the bandit-relevant parameters from `meta_config.json` to `bandit_overrides.json` in the format expected by `HotConfigReloader`. This is the ONLY way the meta-loop modifies the bandit's behavior — it goes through the existing `HotConfigReloader` mechanism, ensuring all validation and logging is applied. Parameters bridged: `T_base`, `exploration_floor`, `paper_preference_ratio`, `K_reheat_threshold`, `reheat_factor`, `min_temperature`, `enable_rollback_safety`. Also writes boost weights as a separate `boost_overrides` section in `bandit_overrides.json` (extending `HotConfigReloader` to support boost weight overrides). |
| **ModelScientistConfigBridge** | Build `meta/bridges/ms_bridge.py`. Class `ModelScientistConfigBridge` with method `apply(meta_config: dict)`. Writes model-scientist-relevant parameters to a `ms_overrides.json` that `ModelScientistPipeline` reads at the start of each iteration cycle. Parameters bridged: `scale_gate_enabled`, `ablation_enabled`, `diagnostics_capture_interval`, `failure_mining_interval`, `metric_correlation_threshold`, `max_active_metrics`, `critic_proposal_frequency`. Extend `ModelScientistPipeline` with a `reload_config()` method (mirroring the bandit's `HotConfigReloader` pattern) that checks for `ms_overrides.json` changes on each iteration. |
| **SurrogateTriageConfigBridge** | Build `meta/bridges/st_bridge.py`. Similar pattern — writes to `st_overrides.json` read by `SurrogateTriagePipeline`. Parameters bridged: `surrogate_retrain_threshold`, `surrogate_training_epochs`, `surrogate_learning_rate`, `surrogate_k_folds`, `cold_start_conservative_threshold`, `cold_start_full_threshold`, `max_queue_size`, `dedup_threshold`, `paper_filter_threshold`, `daily_ingestion_max_results`, `ceiling_monitor_window_size`. |
| **GPUKernelConfigBridge** | Build `meta/bridges/gk_bridge.py`. Similar pattern — writes to `gk_overrides.json` read by `GPUKernelPipeline`. Parameters bridged: `kernel_benchmark_n_warmup`, `kernel_benchmark_n_timed`, `kernel_benchmark_n_runs`, `kernel_evolution_max_generations`, `kernel_evolution_min_improvement`, `kernel_refinement_cooldown`, `kernel_runtime_check_interval`, `kernel_mutations_per_generation`. |

### Subphase 1.3: Safety Boundaries & Sandbox Enforcement

**Status:** NOT IMPLEMENTED
**Gate:** Passes when: (1) the meta-loop sandbox prevents all 5 boundary violations: writing to `train.py`, modifying the validation dataset path, modifying the val_bpb computation, modifying `meta_loop.py` itself, and launching a meta-meta-loop; (2) each violation attempt produces a `BoundaryViolationError` that is logged to `meta_log.jsonl` with the violation type and stack trace; (3) the compute budget enforcer correctly throttles meta-experiments when the 20% budget is exhausted (verified by injecting 100 rapid meta-experiment requests and confirming only 20% are executed).

| Agent | Task |
|-------|------|
| **MetaSandboxEnforcer** | Build `meta/safety/sandbox.py`. Class `MetaSandboxEnforcer` with method `check_write(path: str) → bool`. Maintains a whitelist of writable paths: `meta_config.json`, `bandit_overrides.json`, `ms_overrides.json`, `st_overrides.json`, `gk_overrides.json`, `meta_log.jsonl`, `meta_state.json`, `meta_config_schema.json`, and files in `meta/` and `bandit/prompt_templates/`. Any write to a path not on the whitelist raises `BoundaryViolationError(violation_type="unauthorized_write", path=path)`. The enforcer wraps `open()` and `os.rename()` in the meta-loop's process — implemented as a context manager: `with MetaSandboxEnforcer.enforce(): ...`. Also blocks: `import subprocess` / `os.system` (prevents shelling out to bypass sandbox), modifications to `sys.modules` (prevents monkey-patching evaluation code), and writes to any path containing `train.py`, `eval`, `valid`, or `dataset`. |
| **RecursionDepthGuard** | Build `meta/safety/recursion_guard.py`. Class `RecursionDepthGuard` with method `check_depth() → int`. Reads an environment variable `META_RECURSION_DEPTH` (default 0). The meta-loop sets this to 1 before running. If any process within the meta-loop attempts to launch another meta-loop (detected by checking if `META_RECURSION_DEPTH >= 1`), raise `BoundaryViolationError(violation_type="recursive_meta", depth=current_depth)`. Maximum allowed depth: 1 (the meta-loop itself). This is a hard stop against meta-meta-optimization — even if someone writes code that tries to launch a deeper level, the guard prevents it. |
| **MetaComputeBudgetEnforcer** | Build `meta/safety/compute_budget.py`. Class `MetaComputeBudgetEnforcer` with method `can_run_meta_experiment(total_iterations_run, meta_iterations_run) → bool`. Enforces the configurable budget split (default 80% production, 20% meta-exploration). Track via `meta_state.json`: `{total_inner_iterations, meta_experiment_iterations, budget_fraction: 0.2, budget_exhausted_this_cycle: bool}`. When `meta_iterations / total_iterations > budget_fraction`: return False, log `"budget_exhausted"` to `meta_log.jsonl`, queue the pending meta-experiment for the next cycle. The budget resets every `budget_cycle_length` iterations (default 500). Also implement `get_budget_status() → dict` showing: remaining meta-iterations in current cycle, utilization rate, and projected exhaustion iteration. |
| **EvaluationMetricGuard** | Build `meta/safety/eval_guard.py`. Class `EvaluationMetricGuard` with method `verify_evaluation_unchanged() → bool`. Before each meta-experiment, compute a SHA-256 hash of the evaluation code (the val_bpb computation function in `train.py`'s validation loop + the validation data file path). Store the hash in `meta_state.json`. If the hash changes between meta-experiments, raise `BoundaryViolationError(violation_type="evaluation_modified", hash_before, hash_after)` and abort the meta-experiment. This is the ultimate safety check: even if a bug in the meta-loop accidentally modifies evaluation code, the guard catches it before any meta-experiment uses the corrupted evaluation. |
| **BoundaryViolationTester** | Build `meta/safety/boundary_tester.py`. Class `BoundaryViolationTester` with method `run_all_tests() → list[TestResult]`. Tests: (1) attempt to write to `train.py` → expect `BoundaryViolationError`, (2) attempt to write to a path containing `eval` → expect error, (3) set `META_RECURSION_DEPTH=1` and attempt to launch meta-loop → expect error, (4) exhaust compute budget and attempt another meta-experiment → expect rejection, (5) modify the evaluation hash file and attempt to run → expect error. Each test produces `TestResult(test_name, passed: bool, error_type, detail)`. All 5 must pass for the gate. |

### Subphase 1.4: Baseline Measurement & Statistical Power Analysis

**Status:** NOT IMPLEMENTED
**Gate:** Passes when: (1) 3 independent 100-iteration baseline runs produce val_bpb improvement curves; (2) the improvement rate metric is defined as `IR(window) = Σ(accepted_deltas_in_window) / window_size` computed over rolling windows of size W (default 20); (3) the mean and standard deviation of IR across windows and runs is computed; (4) the minimum detectable effect size (MDES) is computed: the smallest improvement in IR that the meta-loop can detect with 80% power at p < 0.05, given the measured variance and a meta-experiment length of K iterations (default 50).

| Agent | Task |
|-------|------|
| **BaselineRunOrchestrator** | Build `meta/baseline/runner.py`. Class `BaselineRunOrchestrator` with method `run_baselines(n_runs: int = 3, n_iterations: int = 100, seeds: list[int] = [42, 123, 456]) → list[BaselineResult]`. For each seed: (1) checkpoint all four systems' state (bandit, model scientist, surrogate, kernels), (2) run `n_iterations` using `AdaptiveBanditPipeline.run_iteration()` with the current default `meta_config.json`, (3) record every val_bpb delta from `hypothesis_journal.jsonl` entries created during the run (filter by iteration range), (4) restore from checkpoint. `BaselineResult` dataclass: `{seed, n_iterations, deltas: list[float], verdicts: list[str], improvement_rates: list[float] (rolling IR with window=20), mean_ir, std_ir, total_improvement, acceptance_rate}`. The orchestrator must ensure no state leaks between runs — each run starts from the identical checkpoint. |
| **ImprovementRateCalculator** | Build `meta/baseline/improvement_rate.py`. Class `ImprovementRateCalculator` with methods: `compute_rolling(deltas: list[float], window: int = 20) → list[float]` — compute IR at each step as the mean of accepted deltas in the trailing window; `compute_aggregate(baseline_results: list[BaselineResult]) → AggregateIR` — compute cross-run statistics: mean IR, standard deviation, 95% CI, median, IQR, and per-window variance decomposition (within-run variance vs. between-run variance). `AggregateIR` dataclass: `{mean_ir, std_ir, ci_95_lower, ci_95_upper, median_ir, iqr, within_run_variance, between_run_variance, n_windows, n_runs}`. The within-run variance tells us how much IR fluctuates within a single campaign; the between-run variance tells us how much it varies across campaigns with different seeds. The meta-loop needs to exceed the *between-run* variance to claim a real improvement. |
| **MinimumDetectableEffectCalculator** | Build `meta/baseline/power_analysis.py`. Class `MinimumDetectableEffectCalculator` with method `compute_mdes(aggregate_ir: AggregateIR, meta_experiment_length: int = 50, alpha: float = 0.05, power: float = 0.8) → MDESResult`. Uses the standard power analysis formula: `MDES = (z_alpha + z_power) × std_ir × sqrt(2 / n_windows_per_experiment)` where `n_windows_per_experiment = meta_experiment_length / window_size`. `MDESResult` dataclass: `{mdes_absolute: float (absolute IR improvement), mdes_relative: float (MDES / mean_ir), meta_experiment_length, alpha, power, n_windows_per_experiment}`. If `mdes_relative > 0.5` (can only detect > 50% relative improvements): recommend longer meta-experiments or more baseline runs to reduce variance. If `mdes_relative < 0.1` (can detect < 10% improvements): the meta-loop has excellent statistical power — proceed. |
| **MetaExperimentLengthOptimizer** | Build `meta/baseline/experiment_length.py`. Class `MetaExperimentLengthOptimizer` with method `optimize(aggregate_ir: AggregateIR, compute_budget_fraction: float = 0.2, total_iterations_per_cycle: int = 500) → ExperimentLengthResult`. Given the compute budget, determine the optimal meta-experiment length K: longer experiments reduce variance (better MDES) but allow fewer experiments per cycle (less exploration). The tradeoff: `n_experiments_per_cycle = floor(total_iterations × budget_fraction / K)`. The optimal K maximizes expected meta-improvement, which is approximately `n_experiments × P(detect_improvement) × mean_improvement_when_detected`. Use the MDES formula to compute P(detect) as a function of K, and assume mean improvement is the MDES itself (conservative). `ExperimentLengthResult` dataclass: `{optimal_K, n_experiments_per_cycle, mdes_at_optimal_K, total_meta_iterations_per_cycle}`. |

---

## Phase 2: Meta-Bandit Engine & Configuration Variant Generation

**Gate:** Phase 2 is COMPLETE when: (1) the meta-bandit maintains posteriors over at least 10 configuration dimensions, with each dimension tracked by its own `Beta(alpha, beta)` posterior; (2) at least 15 meta-experiments have been run and scored, producing non-trivial posteriors (at least 3 dimensions have posterior mean significantly different from 0.5); (3) at least 1 configuration dimension has been promoted to a non-default value based on statistically significant improvement (one-sided test, p < 0.1); (4) the meta-bandit's arm selection is verified to match Thompson Sampling (chi-squared test on 1000 simulated selections, p > 0.05); (5) all meta-experiments respected the safety sandbox (no boundary violations logged).

### Subphase 2.1: Meta-Bandit Architecture & State Management

**Status:** NOT IMPLEMENTED
**Gate:** Passes when `meta_state.json` contains a Thompson Sampling bandit over configuration dimensions, each dimension has per-variant posteriors, and 5 simulated meta-experiments correctly update the posteriors.

| Agent | Task |
|-------|------|
| **MetaBanditArchitect** | Build `meta/bandit/meta_bandit.py`. Class `MetaBandit` that operates over *configuration dimensions* rather than modification categories (contrasting with the inner `ThompsonSamplerEngine` which operates over arms). Architecture: each dimension has 2-5 discrete variants (e.g., `T_base` has variants `[0.01, 0.02, 0.025, 0.04, 0.05]`). Each variant has a `Beta(alpha, beta)` posterior. Selection: for each dimension, sample from all variants' posteriors, select the variant with the highest sample. This produces a full configuration by independently selecting each dimension's best variant. This is a *parallel bandit* — one bandit per dimension, not one bandit over the Cartesian product of all dimensions (which would be intractable). The parallel assumption (dimension independence) is a simplification; interactions are detected post-hoc by `InteractionDetector` (Phase 3). `MetaBanditState` dataclass: `{dimensions: dict[str, DimensionState], global_meta_iteration: int, meta_regime: str ("baseline"|"active"|"maintenance"), total_meta_experiments: int, budget_used: float, metadata: dict}`. `DimensionState`: `{param_id: str, variants: list, variant_posteriors: dict[str, {alpha: float, beta: float}], current_best: any, last_promoted: float}`. |
| **MetaVariantDiscretizer** | Build `meta/bandit/discretizer.py`. Class `MetaVariantDiscretizer` with method `discretize(param: MetaParameter, n_variants: int = 5) → list`. For continuous parameters: generate `n_variants` evenly spaced values in the valid range, always including the current default. For integer parameters: generate values at quantiles of the range. For boolean parameters: generate `[True, False]`. For special cases: `T_base` variants are log-spaced (temperature scales geometrically): `[0.005, 0.01, 0.025, 0.05, 0.1]`. `exploration_floor` variants: `[0.02, 0.05, 0.08, 0.12, 0.2]`. Prompt template variants are handled separately by `PromptVariantGenerator` (Subphase 2.3). Method `discretize_all(params: list[MetaParameter]) → dict[str, list]` returns the full discretization. |
| **MetaStateManager** | Build `meta/bandit/meta_state.py`. Class `MetaStateManager` with methods `save(state: MetaBanditState, path="meta_state.json")` (atomic write with checksum, same pattern as `AtomicStateManager`), `load(path) → MetaBanditState`, `recover(log_path) → MetaBanditState` (replay from `meta_log.jsonl`). The meta-state is separate from the inner bandit's `strategy_state.json` — the two never interfere. The meta-state records which configuration is currently active (`current_config: dict`), which is the current best (`best_config: dict`), and the full posterior history. |
| **MetaPosteriorUpdater** | Build `meta/bandit/meta_updater.py`. Class `MetaPosteriorUpdater` with method `update(state: MetaBanditState, experiment_result: MetaExperimentResult, baseline_ir: AggregateIR) → MetaBanditState`. For each dimension that was varied in the experiment: (1) compute the experiment's IR using `ImprovementRateCalculator`, (2) if `experiment_IR > baseline_mean_IR + 1_std_IR`: the variant is a "success" → `alpha += 1` for that variant, (3) if `experiment_IR <= baseline_mean_IR - 1_std_IR`: the variant is a "failure" → `beta += 1`, (4) if within ±1 std: inconclusive → `alpha += 0.3, beta += 0.3` (weak update). This three-zone scoring avoids the problem of counting noisy near-baseline results as failures when they might just be unlucky. Log the update to `meta_log.jsonl`. |

### Subphase 2.2: Meta-Experiment Runner & Scoring

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the meta-experiment runner correctly: (1) generates an experimental configuration by sampling from the meta-bandit, (2) applies it to all four systems via the config bridges, (3) runs K inner-loop iterations, (4) computes the improvement rate, (5) compares to baseline, (6) updates the meta-bandit's posteriors, (7) restores the best configuration — all without state corruption in any of the four systems.

| Agent | Task |
|-------|------|
| **MetaExperimentRunner** | Build `meta/experiment/runner.py`. Class `MetaExperimentRunner` with method `run_experiment(meta_state: MetaBanditState, context: MetaContext, experiment_length: int) → MetaExperimentResult`. Steps: (1) sample a configuration from the meta-bandit (one variant per dimension), (2) compute the diff from current_best config, (3) apply the experimental config via all four bridges (`BanditConfigBridge.apply()`, `ModelScientistConfigBridge.apply()`, etc.), (4) run `experiment_length` inner-loop iterations via `AdaptiveBanditPipeline.run_iteration()`, (5) read all journal entries created during the experiment, compute IR, (6) compute secondary metrics: acceptance rate, stepping_stone_rate, arm_selection_entropy, (7) restore the best configuration via bridges, (8) score the experiment against baseline. `MetaExperimentResult` dataclass: `{experiment_id: str, config_diff: list[ParamDiff], n_iterations: int, improvement_rate: float, acceptance_rate: float, stepping_stone_rate: float, entropy: float, compared_to_baseline: str ("better"|"worse"|"inconclusive"), baseline_ir_used: float, raw_deltas: list[float], timestamp: float}`. The experimental config is applied in a way that is fully reversible — if the experiment crashes mid-way, the runner restores the best config in a `finally` block. |
| **MetaExperimentScheduler** | Build `meta/experiment/scheduler.py`. Class `MetaExperimentScheduler` with method `should_run_meta_experiment(meta_state: MetaBanditState, inner_iteration: int, budget_enforcer: MetaComputeBudgetEnforcer) → bool`. Logic: (1) if `meta_regime == "baseline"`: return False (still collecting baselines), (2) if `meta_regime == "maintenance"`: run meta-experiments 5% of the time (reduced exploration), (3) if `meta_regime == "active"`: run meta-experiments at the budgeted rate (20% of iterations), (4) check `budget_enforcer.can_run_meta_experiment()` — if budget exhausted, return False. The scheduler interleaves meta-experiments with production iterations: run K iterations with experimental config (meta-experiment), then run `(1/budget_fraction - 1) × K` iterations with the best config (production), then another meta-experiment. This ensures the production run benefits from the best-known configuration while meta-experiments explore alternatives. |
| **MetaExperimentLogger** | Build `meta/experiment/logger.py`. Class `MetaExperimentLogger` that appends to `meta_log.jsonl`. Entry types: (1) `"meta_experiment_started"`: `{experiment_id, config_diff, experiment_length, meta_iteration}`, (2) `"meta_experiment_completed"`: `{experiment_id, improvement_rate, baseline_ir, verdict, duration_seconds}`, (3) `"meta_posterior_update"`: `{dimension, variant, alpha_before, alpha_after, beta_before, beta_after, experiment_id}`, (4) `"meta_promotion"`: `{dimension, old_value, new_value, evidence: {alpha, beta, posterior_mean, p_value}}`, (5) `"meta_regime_change"`: `{old_regime, new_regime, reason}`, (6) `"meta_budget_status"`: `{total_iterations, meta_iterations, budget_fraction, exhausted}`. |

### Subphase 2.3: Prompt Template Optimization via Evolutionary Generation

**Status:** NOT IMPLEMENTED
**Gate:** Passes when at least 5 prompt template variants per arm have been generated, tested as meta-experiments, and the best-performing variant achieves statistically higher IR than the default for at least 2 of the 7 arm categories (tested via one-sided paired t-test across 3 runs, p < 0.1).

| Agent | Task |
|-------|------|
| **PromptVariantGenerator** | Build `meta/prompts/variant_generator.py`. Class `PromptVariantGenerator` with method `generate_variants(arm_id: str, current_template: str, journal_context: str, n_variants: int = 5) → list[PromptVariant]`. Uses an LLM (the same model used for the inner loop, but called during meta-experiment planning, not during inner-loop execution) to generate prompt variants. Each variant modifies one dimension of the prompt: (1) **Instruction specificity**: range from "Improve the model" (general) to "Reduce attention entropy in layers 1-3 by modifying the normalization approach" (specific — uses diagnostics), (2) **History format**: full diffs vs. summary-only vs. outcomes-only vs. failure-patterns-only, (3) **Code context**: full `train.py` vs. relevant-function-only vs. diff-from-original-only, (4) **Constraint emphasis**: no constraints vs. top-5 constraints vs. all constraints with severity weights, (5) **Reasoning guidance**: no guidance vs. "think step by step" vs. "consider the diagnostic signals first, then propose a targeted change". `PromptVariant` dataclass: `{variant_id, arm_id, template_text, variation_dimension, variation_description, parent_template_hash}`. Generated variants are written to `meta/prompts/candidates/{arm_id}/variant_{i}.txt`. |
| **PromptABEvaluator** | Build `meta/prompts/evaluator.py`. Class `PromptABEvaluator` with method `evaluate(arm_id: str, variant: PromptVariant, experiment_length: int = 50, n_seeds: int = 2) → PromptEvalResult`. For each seed: (1) install the variant template to `bandit/prompt_templates/{arm_id}.txt`, (2) run `experiment_length` iterations (the inner bandit will naturally select this arm sometimes — the prompt only affects that arm's proposals, not all arms), (3) extract the IR contribution from this arm's iterations only (filter journal entries by `bandit_arm:{arm_id}`), (4) restore the default template. `PromptEvalResult`: `{variant_id, arm_id, per_seed_ir: list[float], mean_ir, std_ir, n_arm_selections, n_arm_successes, arm_success_rate}`. Compare to the default template's per-arm IR (from baseline runs) — a variant is "better" if its mean IR exceeds default IR by > 1 std. |
| **PromptEvolutionController** | Build `meta/prompts/evolution.py`. Class `PromptEvolutionController` with method `evolve(arm_id: str, n_generations: int = 3) → PromptVariant`. Evolutionary loop: (1) generate 5 variants from current best, (2) evaluate each, (3) select top-2, (4) generate 5 variants from top-2 (crossover: combine instruction style from one with history format from another), (5) repeat for `n_generations`. Return the best variant across all generations. After evolution, the winner is compared to the default via a final 100-iteration head-to-head test. If the winner is statistically better (p < 0.1), promote it: copy to `bandit/prompt_templates/{arm_id}.txt` and log as `"meta_promotion"` in `meta_log.jsonl`. |

### Subphase 2.4: Context Budget & Evaluation Protocol Optimization

**Status:** NOT IMPLEMENTED
**Gate:** Passes when: (1) at least 3 context allocation strategies have been tested as meta-experiments; (2) at least 3 evaluation protocol variants have been tested; (3) the optimal context allocation and evaluation protocol are identified and promoted (or the defaults confirmed as near-optimal).

| Agent | Task |
|-------|------|
| **ContextBudgetExplorer** | Build `meta/context/budget_explorer.py`. Class `ContextBudgetExplorer` with method `generate_allocations(total_tokens: int = 8000) → list[ContextAllocation]`. Generates allocation variants: (1) `code_heavy`: 60% code, 20% history, 10% diagnostics, 10% constraints, (2) `history_heavy`: 30% code, 40% history, 15% diagnostics, 15% constraints, (3) `diagnostics_heavy`: 30% code, 20% history, 35% diagnostics, 15% constraints, (4) `constraints_heavy`: 30% code, 15% history, 15% diagnostics, 40% constraints, (5) `balanced`: 30% code, 25% history, 25% diagnostics, 20% constraints, (6) `dynamic_early`: early iterations (< 50) get `code_heavy`, later get `history_heavy` (adapts over campaign lifetime). `ContextAllocation`: `{allocation_id, code_fraction, history_fraction, diagnostics_fraction, constraints_fraction, is_dynamic, dynamic_rule: str | None}`. Allocations are applied by modifying `CategoryPromptRouter.build_prompt()` to truncate each section to its token budget. |
| **EvalProtocolExplorer** | Build `meta/evaluation/protocol_explorer.py`. Class `EvalProtocolExplorer` with method `generate_protocols() → list[EvalProtocol]`. Generates evaluation protocol variants: (1) `fast_cheap`: 500 training steps, 1 seed, 10% warmup discard, (2) `standard`: 1000 steps, 1 seed, 20% warmup, (3) `careful`: 1500 steps, 2 seeds, 20% warmup, (4) `thorough`: 2000 steps, 3 seeds, 25% warmup, (5) `two_stage`: fast screening (500 steps, 1 seed) → if promising (delta < 0), full evaluation (1500 steps, 2 seeds). `EvalProtocol`: `{protocol_id, training_steps, n_seeds, warmup_fraction, is_two_stage, stage1_steps, stage1_seeds, stage2_steps, stage2_seeds}`. Each protocol is tested as a meta-experiment by setting `min_training_steps` and `n_eval_seeds` in `meta_config.json`. The tradeoff: cheaper protocols allow more experiments per compute budget but have higher variance in IR measurement. |
| **MetaVarianceCostAnalyzer** | Build `meta/evaluation/variance_cost.py`. Class `MetaVarianceCostAnalyzer` with method `analyze(protocol_results: list[ProtocolEvalResult]) → VarianceCostReport`. For each protocol: (1) compute the variance in IR across repeated identical experiments, (2) compute the wall-clock time per experiment, (3) compute `cost_effectiveness = 1 / (variance × time)`. Find the knee of the curve — the protocol that maximizes information per GPU-hour. Also compute the MDES for each protocol: which is the cheapest protocol that can detect a 10% relative improvement in IR? `VarianceCostReport`: `{per_protocol: dict[str, {variance, time_seconds, cost_effectiveness, mdes}], recommended_protocol_id, recommended_two_stage: bool}`. |

---

## Phase 3: STOP-Style Strategy Generation & Interaction Detection

**Gate:** Phase 3 is COMPLETE when: (1) the STOP scaffold has generated at least 3 novel harness strategies expressed as Python code snippets that plug into defined extension points, and at least 1 strategy differs qualitatively from parameter sweeps (e.g., a new acceptance criterion, a novel prompt chaining approach, or a dynamic scheduling rule); (2) the interaction detector has identified at least 1 parameter interaction (two dimensions whose joint effect differs from the sum of their individual effects) with statistical significance; (3) at least 1 STOP-generated strategy has been promoted to production based on statistically significant improvement.

### Subphase 3.1: STOP Scaffold Implementation

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the scaffold generates a complete harness strategy as a Python code snippet, the `StrategySafetyChecker` validates it, the `StrategyExecutor` runs it in the sandbox for K iterations, and the result is correctly scored.

| Agent | Task |
|-------|------|
| **STOPScaffoldBuilder** | Build `meta/stop/scaffold.py`. Class `STOPScaffold` with method `generate_strategy(current_best_config: dict, meta_experiment_history: list[MetaExperimentResult], baseline_ir: AggregateIR) → GeneratedStrategy`. The scaffold calls an LLM with a structured prompt: "You are optimizing a research automation harness. Here is the current best configuration: {config}. Here are the results of recent meta-experiments: {history}. The baseline improvement rate is {baseline_ir}. Generate a Python code snippet that implements a novel harness strategy. The code snippet must define one of the following hooks: (1) `selection_hook(state, diagnostics) → arm_id` — overrides the Thompson Sampling arm selection, (2) `acceptance_hook(delta, arm_state, diagnostics) → bool` — overrides the annealing acceptance decision, (3) `prompt_hook(arm_id, diagnostics, journal_context) → str` — overrides the prompt construction, (4) `scheduling_hook(iteration, state) → dict` — overrides parameter values for this iteration. The snippet must not import os, subprocess, or any network library. It receives only the approved API objects." `GeneratedStrategy`: `{strategy_id, hook_type, code: str, description: str, llm_rationale: str, estimated_improvement: str}`. |
| **StrategySafetyChecker** | Build `meta/stop/safety_checker.py`. Class `StrategySafetyChecker` with method `check(strategy: GeneratedStrategy) → SafetyCheckResult`. Static analysis: (1) parse the code with `ast.parse()` — reject if syntax error, (2) walk the AST and reject if it contains: `Import` nodes for `os`, `subprocess`, `shutil`, `pathlib`, `socket`, `http`, `urllib`, or any module not in the approved list (`math`, `random`, `numpy`, `scipy.stats`), (3) reject if it contains `open()`, `exec()`, `eval()`, `compile()`, `__import__()` calls, (4) reject if the code exceeds 200 lines (complexity bound), (5) inject a 10-second timeout wrapper — if the hook doesn't return within 10 seconds when called, it's killed and logged as `timeout`. `SafetyCheckResult`: `{safe: bool, violations: list[str], ast_nodes_checked: int}`. |
| **StrategyExecutor** | Build `meta/stop/executor.py`. Class `StrategyExecutor` with method `execute(strategy: GeneratedStrategy, context: MetaContext, experiment_length: int) → MetaExperimentResult`. Steps: (1) verify safety via `StrategySafetyChecker.check()`, (2) compile the code to a callable hook, (3) inject the hook into the inner loop: for `selection_hook`, replace `ThompsonSamplerEngine.select()` temporarily; for `acceptance_hook`, replace `AnnealingAcceptanceEngine.decide()`; for `prompt_hook`, replace `CategoryPromptRouter.build_prompt()`; for `scheduling_hook`, call it at the start of each iteration and write its output to config bridges, (4) run `experiment_length` inner-loop iterations, (5) remove the hook (restore original functions), (6) compute and return `MetaExperimentResult`. The hook injection is done via Python's `unittest.mock.patch` for reversibility. If the hook raises an exception during execution: catch it, log the exception, fall back to the default behavior for that iteration, and continue (don't abort the experiment — partial data is still useful). |
| **StrategyEvolutionController** | Build `meta/stop/evolution.py`. Class `StrategyEvolutionController` with method `evolve(hook_type: str, n_generations: int = 3, n_candidates: int = 3) → GeneratedStrategy`. Evolutionary loop: (1) generate `n_candidates` strategies for `hook_type`, (2) safety-check and execute each, (3) rank by IR, (4) feed the top-2 strategies' code and results back to the LLM with "Improve on these strategies — the first achieved IR={ir1}, the second achieved IR={ir2}. Combine their strengths and fix their weaknesses.", (5) repeat for `n_generations`, (6) return the best strategy across all generations. If the best strategy beats baseline IR + 2σ, promote it by saving the hook code to `meta/active_strategies/{hook_type}.py` and activating it permanently in `meta_config.json`. |

### Subphase 3.2: Parameter Interaction Detection

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the interaction detector identifies at least 1 statistically significant interaction (p < 0.1) from the meta-experiment history, and the joint optimization of the interacting dimensions produces better IR than independent optimization.

| Agent | Task |
|-------|------|
| **InteractionDetector** | Build `meta/interactions/detector.py`. Class `InteractionDetector` with method `detect(experiment_history: list[MetaExperimentResult], significance: float = 0.1) → list[Interaction]`. For every pair of dimensions (i, j) that have been varied in at least 5 experiments: (1) partition experiments into 4 groups: (dim_i=default, dim_j=default), (dim_i=varied, dim_j=default), (dim_i=default, dim_j=varied), (dim_i=varied, dim_j=varied), (2) compute the interaction effect: `interaction = IR(both_varied) - IR(i_only) - IR(j_only) + IR(neither)`, (3) test significance via 2-way ANOVA F-test on the interaction term. `Interaction` dataclass: `{dim_i, dim_j, interaction_effect, p_value, synergy: bool (interaction > 0, the combination is better than sum of parts), antagonism: bool (interaction < 0, the combination is worse)}`. Synergistic interactions suggest the meta-bandit should vary those dimensions together; antagonistic interactions suggest they should be varied independently with caution. |
| **JointOptimizer** | Build `meta/interactions/joint_optimizer.py`. Class `JointOptimizer` with method `optimize_joint(dim_i: str, dim_j: str, experiment_history: list) → tuple`. For an identified interaction: (1) construct the 2D grid of (variant_i, variant_j) pairs, (2) run meta-experiments for unexplored cells of the grid (many may already exist from prior experiments), (3) select the joint optimum — the (variant_i, variant_j) pair with the highest IR. Compare: is the joint optimum better than the independently-optimized (best_i, best_j)? If yes, promote the joint optimum. If no, the interaction is informational only. |

### Subphase 3.3: Compute Budget Optimization & ROI Tracking

**Status:** NOT IMPLEMENTED
**Gate:** Passes when: (1) the meta-exploration budget is tracked accurately (within ±5% of target), (2) the ROI tracker shows cumulative improvement attributable to meta-optimization, (3) the budget has been dynamically adjusted at least once based on meta-performance (e.g., increased from 20% to 25% if meta-experiments are consistently producing improvements, or decreased to 10% if they're not).

| Agent | Task |
|-------|------|
| **MetaBudgetOptimizer** | Build `meta/budget/optimizer.py`. Class `MetaBudgetOptimizer` with method `recommend_budget(meta_state: MetaBanditState, roi_data: ROIData) → BudgetRecommendation`. Logic: (1) if the last 5 meta-experiments all produced improvements: `recommended_fraction = min(0.35, current_fraction × 1.25)` — increase exploration, (2) if the last 5 all produced no improvement: `recommended_fraction = max(0.05, current_fraction × 0.6)` — decrease exploration and conserve compute, (3) if mixed: keep current fraction. `BudgetRecommendation`: `{current_fraction, recommended_fraction, reason, confidence}`. The recommendation is written to `meta_config.json` only if `enable_auto_budget: True` (default False — requires operator opt-in). |
| **MetaROITracker** | Build `meta/budget/roi_tracker.py`. Class `MetaROITracker` with method `compute_roi(meta_state: MetaBanditState, experiment_history: list, baseline_ir: AggregateIR) → ROIData`. Tracks: (1) `total_meta_iterations`: iterations spent on meta-experiments, (2) `total_production_iterations`: iterations spent on best-known config, (3) `improvement_from_meta`: the difference in IR between the current promoted config and the original defaults, multiplied by the number of production iterations (total improvement attributable to meta-tuning), (4) `cost_of_meta`: total_meta_iterations (these iterations produced data but at an experimental config, so their improvement rate may be lower than production), (5) `roi = improvement_from_meta / cost_of_meta`. `ROIData`: `{total_meta_iterations, total_production_iterations, improvement_from_meta, cost_of_meta, roi, cumulative_val_bpb_improvement, attribution: dict[str, float] (per-dimension contribution)}`. |

---

## Phase 4: Convergence Detection, Regime Management & Configuration Documentation

**Gate:** Phase 4 is COMPLETE when: (1) the convergence detector correctly identifies convergence (no promoted changes in 5 consecutive meta-cycles, meta-bandit posteriors have low variance on all dimensions); (2) the system enters maintenance mode with reduced exploration (5% budget), and the maintenance mode produces stable performance (IR within ±10% of the converged configuration's mean IR over 200 iterations); (3) the divergence watcher correctly re-triggers active exploration when IR drops significantly (injected by temporarily applying a bad configuration); (4) the converged configuration is documented with per-dimension evidence in `meta_config_report.json`; (5) at least 3 transferable insights have been extracted and validated.

### Subphase 4.1: Convergence Detection & Maintenance Mode

**Status:** NOT IMPLEMENTED
**Gate:** Passes when: (1) convergence is detected after 50+ meta-experiments with no promotion in 5 consecutive cycles, (2) maintenance mode reduces meta-budget to 5%, (3) the system correctly re-enters active mode when IR drops below `baseline_mean - 2σ` for 3 consecutive windows.

| Agent | Task |
|-------|------|
| **MetaConvergenceDetector** | Build `meta/convergence/detector.py`. Class `MetaConvergenceDetector` with method `check(meta_state: MetaBanditState, promotion_history: list) → ConvergenceStatus`. Convergence criteria: (1) no configuration promotion in the last `convergence_window` meta-experiments (default 5), (2) all dimension posteriors have `posterior_variance < 0.01` (well-characterized), (3) the best config's IR is statistically indistinguishable from the last 10 meta-experiments' IRs (F-test, p > 0.1 — variance between experiments is not greater than within-experiment variance). `ConvergenceStatus`: `{converged: bool, meta_experiments_since_last_promotion, max_posterior_variance, f_test_p_value, recommendation: "continue"|"enter_maintenance"|"already_in_maintenance"}`. |
| **MaintenanceModeManager** | Build `meta/convergence/maintenance.py`. Class `MaintenanceModeManager` with method `enter_maintenance(meta_state: MetaBanditState, meta_config: dict) → MetaBanditState`. Actions: (1) set `meta_regime: "maintenance"`, (2) reduce `budget_fraction` to 0.05 (5%), (3) log the transition to `meta_log.jsonl` as `"meta_regime_change"` with `reason: "convergence_detected"`, (4) freeze the best config — all 30+ parameters are locked to their promoted values, (5) continue light exploration: every 100 production iterations, run 1 meta-experiment varying a random dimension to check for regime changes. In maintenance mode, the meta-bandit's posteriors continue to update but promotions require stronger evidence: p < 0.01 (instead of 0.1) to promote a change while in maintenance. |
| **DivergenceWatcher** | Build `meta/convergence/divergence.py`. Class `DivergenceWatcher` with method `check(meta_state: MetaBanditState, recent_ir_windows: list[float], baseline_ir: AggregateIR) → DivergenceAlert | None`. Monitors the production run's IR: if the rolling IR drops below `baseline_mean - 2σ` for 3 consecutive windows of 20 iterations each: emit `DivergenceAlert`: `{triggered: True, current_ir, baseline_ir, drop_magnitude, windows_below_threshold: 3, recommendation: "re_enter_active"}`. The manager then: (1) sets `meta_regime: "active"`, (2) restores `budget_fraction` to 0.20, (3) logs the transition, (4) re-opens all dimension posteriors for exploration (soft-reset: multiply all alpha and beta by 0.5, preserving the mean but widening variance to reflect increased uncertainty). This handles the case where the optimization landscape has shifted (e.g., the model improved enough that different harness parameters are now optimal). |

### Subphase 4.2: Configuration Documentation & Sensitivity Analysis

**Status:** NOT IMPLEMENTED
**Gate:** Passes when `meta_config_report.json` contains per-dimension: the promoted value, the alternatives tested, the IR under each alternative, the p-value for the promotion decision, and a plain-English justification. The sensitivity analysis identifies which dimensions are critical (perturbation causes > 10% IR change) and which are robust.

| Agent | Task |
|-------|------|
| **MetaConfigDocumenter** | Build `meta/documentation/config_documenter.py`. Class `MetaConfigDocumenter` with method `document(meta_state: MetaBanditState, experiment_history: list, baseline_ir: AggregateIR) → ConfigDocumentation`. For each of the 30+ dimensions: (1) `promoted_value`: the current best variant, (2) `default_value`: the original default, (3) `all_variants_tested`: list of variants with their IRs and posterior means, (4) `promotion_evidence`: the meta-experiment(s) that led to promotion, with IR values and p-value, (5) `justification`: a generated plain-English explanation ("T_base was increased from 0.025 to 0.04 because higher temperature improved the stepping-stone rate from 8% to 15%, leading to better exploration of the optimizer arm — 3 meta-experiments confirmed this with p=0.07"), (6) `sensitivity`: how much IR changes with ±10% perturbation (from `SensitivityAnalyzer`). `ConfigDocumentation`: `{dimensions: dict[str, DimensionDoc], total_experiments, total_promotions, best_config_ir, default_config_ir, improvement_over_defaults}`. Output to `meta_config_report.json`. |
| **MetaSensitivityAnalyzer** | Build `meta/documentation/sensitivity.py`. Class `MetaSensitivityAnalyzer` with method `analyze(meta_state, experiment_history, n_perturbations: int = 3) → SensitivityReport`. For each dimension: (1) perturb the promoted value by +10%, (2) run a meta-experiment, (3) perturb by -10%, (4) run a meta-experiment, (5) compute `sensitivity = |IR_+10% - IR_-10%| / (0.2 × IR_promoted)`. Classify: `sensitivity > 0.3` → "critical" (small changes matter), `sensitivity ∈ [0.1, 0.3]` → "moderate", `sensitivity < 0.1` → "robust" (changes don't matter much). `SensitivityReport`: `{per_dimension: dict[str, {perturbation_plus_ir, perturbation_minus_ir, sensitivity, classification}], critical_dimensions: list, robust_dimensions: list}`. Critical dimensions should be monitored closely; robust dimensions can be safely ignored. |

### Subphase 4.3: Meta-Learning Knowledge Transfer & Validation

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when at least 3 transferable insights are extracted from the meta-optimization history, validated on a second campaign (different model scale or dataset), and compiled into `meta_knowledge_base.json`.

| Agent | Task |
|-------|------|
| **InsightExtractor** | Build `meta/knowledge/insight_extractor.py`. Class `InsightExtractor` with method `extract(experiment_history: list, config_documentation: ConfigDocumentation, sensitivity_report: SensitivityReport) → list[MetaInsight]`. Analyze the full meta-optimization history to extract generalizable patterns: (1) **Universal insights**: parameters where the optimal value is consistently the same regardless of inner-loop progress (e.g., "T_base should be 0.04 from the start" — this would transfer to any campaign), (2) **Phase-dependent insights**: parameters whose optimal value changes during the campaign (e.g., "exploration_floor should be 0.12 for the first 100 iterations, then 0.05 after" — transfers as a scheduling rule), (3) **Scale-dependent insights**: parameters that depend on model/data scale (e.g., "kernel_benchmark_n_runs should be 3 for models < 100M params, 5 for larger" — transfers as a conditional rule), (4) **Interaction insights**: identified parameter interactions (e.g., "T_base and exploration_floor interact synergistically — increase both together"). Each insight includes: the evidence (which experiments support it), the confidence (how many data points, how significant), and the transferability assessment (universal vs. phase-dependent vs. scale-dependent). `MetaInsight`: `{insight_id, type: str, description: str, evidence_experiments: list[str], confidence: str, transferability: str, recommended_default: any}`. |
| **TransferValidator** | Build `meta/knowledge/transfer_validator.py`. Class `TransferValidator` with method `validate(insights: list[MetaInsight], validation_campaign_context: MetaContext) → list[TransferValidationResult]`. For each insight: (1) apply the insight's recommended default to the validation campaign's `meta_config.json`, (2) run 100 inner-loop iterations, (3) compare IR to the validation campaign's default config IR, (4) if the insight improves IR: `validated: True`, (5) if not: `validated: False` — the insight is campaign-specific, not transferable. `TransferValidationResult`: `{insight_id, validated: bool, validation_ir, default_ir, improvement: float, p_value: float}`. Insights that validate on 2+ campaigns are marked as "high-confidence transferable" and included in the knowledge base. |
| **MetaKnowledgeBaseWriter** | Build `meta/knowledge/knowledge_base.py`. Class `MetaKnowledgeBaseWriter` with method `compile(validated_insights: list, experiment_summary: dict, config_documentation: ConfigDocumentation) → dict`. Produces `meta_knowledge_base.json` containing: (1) **Validated universal insights** (apply to any campaign), (2) **Validated conditional insights** (apply under specified conditions), (3) **Campaign-specific findings** (interesting but not validated for transfer), (4) **Parameter interaction map** (which dimensions interact), (5) **Sensitivity classification** (which parameters matter), (6) **Recommended defaults** (for starting a new campaign — the meta-optimized configuration), (7) **Anti-patterns** (configurations that consistently underperform — "never set exploration_floor below 0.02" with evidence). This is the lasting artifact of meta-autoresearch: not just a tuned config but empirical knowledge about how to configure the four-system pipeline. |

---

## Phase 5: Production Hardening, Dashboard & Long-Term Monitoring

**Gate:** Phase 5 is COMPLETE when: (1) the meta-loop runs for 500 production iterations (with interleaved meta-experiments) without any safety boundary violations, state corruption, or inner-loop degradation; (2) the unified five-system dashboard shows meta-bandit posteriors, meta-experiment history, ROI, and the active configuration alongside all four inner-system dashboards; (3) the converged configuration is promoted and the system operates in stable maintenance mode; (4) a 200-iteration head-to-head comparison shows the meta-optimized configuration achieves statistically better IR than the original defaults (p < 0.05); (5) the `meta_knowledge_base.json` contains at least 3 validated transferable insights.

### Subphase 5.1: Extended Validation & Safety Verification

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when 500 production iterations complete with: (1) no `BoundaryViolationError` in `meta_log.jsonl`, (2) `meta_state.json` checksum verifies at every iteration, (3) inner-loop state (`strategy_state.json`, `hypothesis_journal.jsonl`, `kernel_config.json`) is not corrupted, (4) evaluation metric hash is unchanged from baseline, (5) the meta-loop has correctly managed the compute budget (actual meta-fraction within ±5% of target).

| Agent | Task |
|-------|------|
| **MetaExtendedValidator** | Build `meta/validation/extended_validator.py`. Class `MetaExtendedValidator` with method `validate(n_iterations: int = 500) → ExtendedMetaValidationResult`. Runs the full five-system pipeline for `n_iterations`: inner bandit loop + meta-experiments at the budgeted rate. At every iteration: (1) verify `MetaSandboxEnforcer` has no violations, (2) verify `EvaluationMetricGuard.verify_evaluation_unchanged()`, (3) verify `AtomicStateManager.detect_corruption()` returns False for all state files, (4) verify `MetaComputeBudgetEnforcer.get_budget_status()` is within bounds. At every 100th iteration: (5) run `BanditHealthAuditor.audit()` and `CrossSystemHealthAuditor.audit()` and `BanditHealthAuditor.audit()`, (6) verify all pass. `ExtendedMetaValidationResult`: `{passed: bool, n_iterations, boundary_violations: int, state_corruptions: int, eval_hash_changes: int, budget_deviation_percent: float, health_audit_failures: int, meta_experiments_run: int, promotions: int}`. |
| **DefaultsVsMetaComparator** | Build `meta/validation/defaults_comparison.py`. Class `DefaultsVsMetaComparator` with method `compare(n_iterations: int = 200, n_seeds: int = 3) → ComparisonResult`. Rigorous head-to-head: for each seed, run `n_iterations` with the meta-optimized config (treatment) and `n_iterations` with the original defaults (control). Same starting model, same data, same seeds. Compute: (1) cumulative val_bpb improvement for treatment vs. control, (2) Mann-Whitney U test (one-sided, treatment >= control), (3) per-dimension contribution: for each promoted dimension, what fraction of the improvement is attributable to that change? (Estimated by running with only that dimension changed and comparing to the full meta-config.) `ComparisonResult`: `{treatment_median_improvement, control_median_improvement, u_statistic, p_value, significant: bool, per_dimension_contributions: dict, effect_size, verdict: str}`. |

### Subphase 5.2: Five-System Unified Dashboard

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when the HTML dashboard displays all 5 systems in a tabbed view, the meta-optimization section shows meta-bandit posteriors, meta-experiment history, ROI curve, and the active configuration diff from defaults.

| Agent | Task |
|-------|------|
| **MetaDashboardBuilder** | Build `meta/dashboard.py`. Class `MetaDashboard` with methods `render_cli(meta_state, roi_data, experiment_history) → str` and `render_html(...) → str`. Sections: (1) **Meta-Regime & Budget**: current regime (baseline/active/maintenance), budget fraction, utilization, ROI, (2) **Dimension Posteriors**: per-dimension table showing current_best, posterior_mean, 95% CI, promotion status, sensitivity, (3) **Meta-Experiment History**: last 20 experiments with config_diff, IR, verdict, (4) **Active Configuration**: full diff between current config and original defaults, highlighted promotions, (5) **STOP Strategies**: any active strategies, their hook types, and performance metrics, (6) **Convergence Status**: meta-experiments since last promotion, posterior variances, convergence predicted at iteration N, (7) **Knowledge Base Summary**: validated insights count, transfer validation status. |
| **FiveSystemDashboardIntegrator** | Extend `PipelineMonitor` in `model_scientist/integration/monitor.py` to aggregate all 5 dashboards: Model Scientist, Surrogate Triage, GPU Kernels, Bandit, and Meta-Optimization. The top-level view has 5 tabs. The summary bar adds: `meta_regime`, `meta_experiments_run`, `meta_roi`, `total_promotions`. This is the final dashboard — the operator sees the entire autoresearch stack in one view, from meta-optimization (top level) down to individual kernel speedups (bottom level). |

### Subphase 5.3: Long-Term Monitoring & Knowledge Base Maintenance

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when: (1) the divergence watcher has been active for 500+ production iterations without false positives, (2) the knowledge base has been updated at least once based on new validated insights from the maintenance-mode exploration, (3) a second autoresearch campaign has been initialized using the knowledge base's recommended defaults and achieved faster time-to-first-improvement than a campaign using original defaults.

| Agent | Task |
|-------|------|
| **LongTermStabilityMonitor** | Build `meta/monitoring/stability.py`. Class `LongTermStabilityMonitor` with method `check(meta_state, recent_ir_windows: list[float], divergence_watcher_history: list) → StabilityReport`. Monitors: (1) IR stability: is the rolling IR within ±10% of the converged configuration's mean IR? (2) False positive rate: how many times has `DivergenceWatcher` triggered when the IR recovered without intervention? (3) Meta-experiment quality: in maintenance mode, are the light exploration experiments producing meaningful information or just confirming the status quo? `StabilityReport`: `{ir_stable: bool, mean_ir, std_ir, divergence_triggers: int, false_positives: int, maintenance_experiments: int, maintenance_discoveries: int}`. If `maintenance_discoveries > 0`: the optimization landscape has shifted, and the meta-loop correctly detected it via maintenance exploration. |
| **KnowledgeBaseUpdater** | Build `meta/knowledge/updater.py`. Class `KnowledgeBaseUpdater` with method `update(knowledge_base: dict, new_insights: list[MetaInsight], new_validations: list[TransferValidationResult]) → dict`. Called periodically (every 100 maintenance iterations) to check if new meta-experiments have produced insights that should be added to or modify the knowledge base. New insights that validate replace older insights with lower confidence. Insights that are invalidated (previously validated but now failing on new data) are moved to a `"deprecated_insights"` section with the reason for deprecation. The knowledge base is a living document — it improves as the system accumulates more experience across campaigns. |
| **NewCampaignBootstrapper** | Build `meta/knowledge/bootstrapper.py`. Class `NewCampaignBootstrapper` with method `bootstrap(knowledge_base_path: str, new_campaign_dir: str) → dict`. Initializes a new autoresearch campaign using the knowledge base's recommended defaults: (1) read `meta_knowledge_base.json`, (2) extract universal insights and apply their recommended defaults to `meta_config.json`, (3) apply conditional insights based on the new campaign's characteristics (model size, dataset, hardware — specified in a `campaign_profile.json`), (4) set `meta_regime: "active"` (the knowledge base provides a warm start, but the meta-loop should still explore to adapt to the new campaign), (5) warm-start the meta-bandit's posteriors from the knowledge base's posterior snapshots (transfer learning: the new campaign starts with informative priors instead of `Beta(1,1)` on each dimension). Return the bootstrapped `meta_config.json`. |

---

## Full Pipeline Integration

**Gate:** The Meta-Autoresearch system is **IMPLEMENTED** when all five phases pass their gates AND the following end-to-end criteria are met simultaneously:

1. The meta-optimized configuration produces statistically better improvement rate than the original defaults (p < 0.05, 200-iteration head-to-head, 3 seeds)
2. At least 1 STOP-generated strategy has been promoted to production (generating genuinely novel harness logic, not just parameter tuning)
3. Safety boundaries held throughout 500+ iterations — no evaluation metric tampering, no recursive meta-loops, no budget overruns > 5%
4. The meta-knowledge base contains at least 3 validated, transferable insights
5. The system has demonstrated convergence and operates in stable maintenance mode with 5% exploration budget
6. At least 1 parameter interaction has been identified and jointly optimized
7. The ROI of meta-optimization is positive: `improvement_from_meta > cost_of_meta` (the iterations spent on meta-experiments produced enough knowledge to more than compensate for their cost)
8. A second campaign bootstrapped from the knowledge base achieves faster time-to-first-improvement than a campaign using original defaults
9. The five-system unified dashboard displays all components in a single view
10. All four config bridges are functional: changes to `meta_config.json` propagate to all four systems within 1 iteration

| Agent | Task |
|-------|------|
| **FullMetaPipelineValidator** | Run the complete five-system stack (Model Scientist + Surrogate Triage + GPU Kernels + Bandit + Meta-Optimization) for 500 inner-loop iterations with interleaved meta-experiments. Verify: all safety boundaries held, all meta-experiments scored correctly, at least 1 configuration was promoted, convergence was detected, maintenance mode was entered, the divergence watcher is armed, and the final configuration outperforms defaults. |
| **MetaImpactQuantifier** | Produce the definitive meta-optimization impact report. Compute: (1) total val_bpb improvement with meta-optimization vs. without, (2) per-dimension contribution (which promoted parameters contributed most?), (3) per-system contribution (did meta-tuning of bandit parameters help more than meta-tuning of surrogate parameters?), (4) STOP strategy contribution (how much of the improvement came from generated strategies vs. parameter tuning?), (5) compute cost: total GPU-hours on meta-experiments vs. the improvement they produced. Format as `meta_impact_report.json`. |
| **MetaDocumentationAgent** | Write comprehensive operator documentation in `meta/README.md` covering: (1) what meta-autoresearch does and why (optimize the harness, not the model), (2) the mathematical safety argument (arXiv:2601.05280 entropy decay, Goodhart's Law at the meta level, why evaluation must be fixed), (3) the 30+ meta-parameters with their defaults, ranges, and sensitivity classifications, (4) how to interpret the meta-dashboard, (5) how to override meta-decisions (`meta_config.json` manual edits), (6) how to add new meta-parameters (extend the inventory, add to the schema, create a bridge), (7) how to start a new campaign from the knowledge base, (8) troubleshooting: what to do when meta-experiments are all failing (probably a baseline shift — check divergence watcher), when the meta-loop isn't converging (increase experiment length or reduce dimensions), when safety boundaries are triggered (never disable them — find the bug), (9) the STOP scaffold: how to write custom hook code, the approved API, the safety checks. |

---

## Agent Summary

| Agent Name | Phase | Subphase | Role |
|---|---|---|---|
| BanditParameterInventorist | 1 | 1.1 | Enumerate 15+ bandit meta-parameters with code paths |
| ModelScientistParameterInventorist | 1 | 1.1 | Enumerate 10+ model scientist meta-parameters |
| SurrogateTriageParameterInventorist | 1 | 1.1 | Enumerate 11+ surrogate triage meta-parameters |
| GPUKernelParameterInventorist | 1 | 1.1 | Enumerate 9+ GPU kernel meta-parameters |
| MetaConfigSchemaBuilder | 1 | 1.1 | Build unified JSON Schema for all 30+ parameters |
| MetaConfigManager | 1 | 1.2 | Load/save/validate unified `meta_config.json` |
| BanditConfigBridge | 1 | 1.2 | Bridge meta-config to `bandit_overrides.json` |
| ModelScientistConfigBridge | 1 | 1.2 | Bridge meta-config to `ms_overrides.json` |
| SurrogateTriageConfigBridge | 1 | 1.2 | Bridge meta-config to `st_overrides.json` |
| GPUKernelConfigBridge | 1 | 1.2 | Bridge meta-config to `gk_overrides.json` |
| MetaSandboxEnforcer | 1 | 1.3 | Whitelist-based file write guard for meta-loop |
| RecursionDepthGuard | 1 | 1.3 | Hard stop against meta-meta-optimization |
| MetaComputeBudgetEnforcer | 1 | 1.3 | Enforce 80/20 compute split with cycle tracking |
| EvaluationMetricGuard | 1 | 1.3 | SHA-256 hash verification of evaluation code |
| BoundaryViolationTester | 1 | 1.3 | Adversarial tests for all 5 safety boundaries |
| BaselineRunOrchestrator | 1 | 1.4 | Run 3×100-iteration baseline campaigns |
| ImprovementRateCalculator | 1 | 1.4 | Compute rolling IR with cross-run statistics |
| MinimumDetectableEffectCalculator | 1 | 1.4 | Power analysis for meta-experiment design |
| MetaExperimentLengthOptimizer | 1 | 1.4 | Determine optimal K for budget/power tradeoff |
| MetaBanditArchitect | 2 | 2.1 | Design parallel per-dimension Thompson Sampling |
| MetaVariantDiscretizer | 2 | 2.1 | Discretize continuous parameters into testable variants |
| MetaStateManager | 2 | 2.1 | Atomic meta-state persistence with recovery |
| MetaPosteriorUpdater | 2 | 2.1 | Three-zone (success/failure/inconclusive) posterior updates |
| MetaExperimentRunner | 2 | 2.2 | Run K inner-loop iterations with experimental config |
| MetaExperimentScheduler | 2 | 2.2 | Interleave meta-experiments with production iterations |
| MetaExperimentLogger | 2 | 2.2 | Append-only JSONL log for meta-events |
| PromptVariantGenerator | 2 | 2.3 | LLM-powered prompt template variant generation |
| PromptABEvaluator | 2 | 2.3 | Per-arm prompt variant A/B evaluation |
| PromptEvolutionController | 2 | 2.3 | Multi-generation evolutionary prompt optimization |
| ContextBudgetExplorer | 2 | 2.4 | Generate context token allocation strategies |
| EvalProtocolExplorer | 2 | 2.4 | Generate evaluation protocol variants |
| MetaVarianceCostAnalyzer | 2 | 2.4 | Cost-effectiveness analysis of evaluation protocols |
| STOPScaffoldBuilder | 3 | 3.1 | LLM-generated harness strategy code snippets |
| StrategySafetyChecker | 3 | 3.1 | AST-based static analysis of generated code |
| StrategyExecutor | 3 | 3.1 | Sandboxed hook injection and execution |
| StrategyEvolutionController | 3 | 3.1 | Multi-generation evolutionary strategy optimization |
| InteractionDetector | 3 | 3.2 | 2-way ANOVA for parameter interaction effects |
| JointOptimizer | 3 | 3.2 | Grid-search joint optimization of interacting dimensions |
| MetaBudgetOptimizer | 3 | 3.3 | Dynamic compute budget adjustment |
| MetaROITracker | 3 | 3.3 | Return on meta-investment tracking |
| MetaConvergenceDetector | 4 | 4.1 | Detect convergence via promotion history + posterior variance |
| MaintenanceModeManager | 4 | 4.1 | Reduce exploration budget and freeze best config |
| DivergenceWatcher | 4 | 4.1 | Re-trigger active exploration on IR drop |
| MetaConfigDocumenter | 4 | 4.2 | Per-dimension evidence-based documentation |
| MetaSensitivityAnalyzer | 4 | 4.2 | ±10% perturbation sensitivity classification |
| InsightExtractor | 4 | 4.3 | Extract universal/conditional/scale-dependent insights |
| TransferValidator | 4 | 4.3 | Validate insight transfer to new campaigns |
| MetaKnowledgeBaseWriter | 4 | 4.3 | Compile validated insights into living knowledge base |
| MetaExtendedValidator | 5 | 5.1 | 500-iteration five-system validation with safety checks |
| DefaultsVsMetaComparator | 5 | 5.1 | 200-iteration head-to-head vs. original defaults |
| MetaDashboardBuilder | 5 | 5.2 | CLI + HTML meta-optimization dashboard |
| FiveSystemDashboardIntegrator | 5 | 5.2 | Unified five-system tabbed HTML dashboard |
| LongTermStabilityMonitor | 5 | 5.3 | IR stability and false-positive monitoring |
| KnowledgeBaseUpdater | 5 | 5.3 | Periodic knowledge base refresh from new data |
| NewCampaignBootstrapper | 5 | 5.3 | Initialize new campaigns from knowledge base |
| FullMetaPipelineValidator | Integration | — | 500-iteration five-system end-to-end validation |
| MetaImpactQuantifier | Integration | — | Definitive meta-optimization impact measurement |
| MetaDocumentationAgent | Integration | — | Comprehensive documentation with safety proofs |

---

## Cross-Reference: Five-System Data Flow

```
 ┌──────────────────────────────────────────────────────────────────────────────────────┐
 │                        META-AUTORESEARCH (THIS PLAN)                                 │
 │                                                                                      │
 │  meta_config.json        ◄── Unified 30+ parameter config                           │
 │  meta_config_schema.json ◄── JSON Schema for validation                             │
 │  meta_state.json         ◄── Meta-bandit posteriors per dimension                   │
 │  meta_log.jsonl          ◄── Meta-experiment log                                    │
 │  meta_knowledge_base.json ◄── Validated transferable insights                       │
 │  meta_config_report.json ◄── Per-dimension evidence documentation                  │
 │                                                                                      │
 │  meta_config.json ─────────────────────────────────────────────────────────────┐     │
 │       │                    │                     │                     │        │     │
 │       ▼                    ▼                     ▼                     ▼        │     │
 │  BanditConfigBridge   MSConfigBridge     STConfigBridge      GKConfigBridge     │     │
 │       │                    │                     │                     │        │     │
 │       ▼                    ▼                     ▼                     ▼        │     │
 │  bandit_overrides.json ms_overrides.json st_overrides.json gk_overrides.json   │     │
 │       │                    │                     │                     │        │     │
 │       ▼                    ▼                     ▼                     ▼        │     │
 │  ┌─────────┐    ┌──────────────┐    ┌────────────────┐    ┌────────────────┐   │     │
 │  │ BANDIT  │    │ MODEL        │    │ SURROGATE      │    │ GPU KERNELS    │   │     │
 │  │         │    │ SCIENTIST    │    │ TRIAGE         │    │                │   │     │
 │  │ T_base  │    │ diagnostics  │    │ retrain_thresh │    │ benchmark_runs │   │     │
 │  │ expl_   │    │ _interval    │    │ filter_thresh  │    │ evolution_gens │   │     │
 │  │ floor   │    │ ablation_    │    │ queue_size     │    │ check_interval │   │     │
 │  │ paper_  │    │ enabled      │    │ cold_start     │    │ mutation_count │   │     │
 │  │ pref    │    │ critic_freq  │    │ ceiling_window │    │ cooldown_secs  │   │     │
 │  │ reheat  │    │ metric_prune │    │ ingestion_max  │    │ min_improve    │   │     │
 │  │ boost_  │    │ eval_seeds   │    │ surrogate_lr   │    │ block_sizes    │   │     │
 │  │ weights │    │ training_    │    │ surrogate_     │    │                │   │     │
 │  │         │    │ steps        │    │ epochs         │    │                │   │     │
 │  └────┬────┘    └──────┬───────┘    └───────┬────────┘    └────────┬───────┘   │     │
 │       │                │                     │                     │            │     │
 │       └────────────────┴─────────────────────┴─────────────────────┘            │     │
 │                                    │                                            │     │
 │                                    ▼                                            │     │
 │                    hypothesis_journal.jsonl  (shared ground truth)              │     │
 │                                    │                                            │     │
 │                                    ▼                                            │     │
 │              ImprovementRateCalculator.compute_rolling()                        │     │
 │                                    │                                            │     │
 │                                    ▼                                            │     │
 │                    MetaPosteriorUpdater.update()                                │     │
 │                                    │                                            │     │
 │                                    ▼                                            │     │
 │                    meta_state.json (posteriors evolve)                          │     │
 │                                                                                │     │
 │  SAFETY ENVELOPE:                                                              │     │
 │  ┌───────────────────────────────────────────────────────────────────────────┐  │     │
 │  │ MetaSandboxEnforcer    — whitelist writes to override files only         │  │     │
 │  │ RecursionDepthGuard    — META_RECURSION_DEPTH == 1, never deeper        │  │     │
 │  │ MetaComputeBudgetEnforcer — 80% production / 20% exploration            │  │     │
 │  │ EvaluationMetricGuard  — SHA-256 of val_bpb code, immutable            │  │     │
 │  └───────────────────────────────────────────────────────────────────────────┘  │     │
 │                                                                                │     │
 │  STOP SCAFFOLD:                                                                │     │
 │  ┌───────────────────────────────────────────────────────────────────────────┐  │     │
 │  │ Generated hook code → StrategySafetyChecker (AST analysis)              │  │     │
 │  │                     → StrategyExecutor (sandboxed injection)             │  │     │
 │  │                     → Scored by ImprovementRateCalculator               │  │     │
 │  │                     → Promoted to meta/active_strategies/ if better      │  │     │
 │  └───────────────────────────────────────────────────────────────────────────┘  │     │
 │                                                                                │     │
 └────────────────────────────────────────────────────────────────────────────────┘     │
```
