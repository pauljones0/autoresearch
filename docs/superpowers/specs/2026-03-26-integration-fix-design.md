# AutoResearch Multi-Layer Integration Fix — Design Spec

**Date:** 2026-03-26
**Approach:** Bottom-up (fix signatures → smoke tests → intra-layer wiring → cross-layer integration → entry point)

## Context

AutoResearch is Karpathy's autonomous ML research system: an AI agent edits `train.py`, trains for 5 minutes, checks if `val_bpb` improved, keeps or discards, loops forever. Five layers have been built on top to upgrade this loop into a principled research organization:

- **bandit/** — Thompson Sampling over modification categories + per-arm simulated annealing for accept/reject
- **surrogate_triage/** — arXiv paper ingestion → surrogate MLP scoring → candidate funnel → GPU evaluation
- **model_scientist/** — diagnostics, ablation, scaling law predictions, metric evolution
- **gpu_kernels/** — Triton kernel generation, verification, benchmarking, evolutionary refinement
- **meta/** — meta-optimization of ~45 harness parameters across the 4 sub-layers

A 6-agent code review found **27 critical bugs, 41 important issues, and zero tests** across 40K lines. The layers are individually well-architected but have never been executed — every pipeline crashes on instantiation due to constructor/method signature mismatches, and the cross-layer wiring is entirely dead code.

## Architecture Overview

```
meta/ (tunes parameters of everything below via Thompson Sampling)
 |-- writes {layer}_overrides.json files
 |-- reads hypothesis_journal.jsonl for improvement rate scoring
 |
 +-- bandit/ (decides what modification category to try + whether to accept)
      |-- dispatches to model_scientist/ (architecture/optimizer/hyperparam arms)
      |-- dispatches to surrogate_triage/ (paper-sourced arms)
      |-- dispatches to gpu_kernels/ (kernel arms)
      |
      +-- model_scientist/ evaluates modifications with scientific rigor
      +-- surrogate_triage/ pre-screens paper candidates, feeds winners to model_scientist
      +-- gpu_kernels/ generates custom Triton kernels for training speedup
```

All layers share a common data contract: `hypothesis_journal.jsonl` records every experiment. The fixed ground truth metric is `val_bpb` from `prepare.py`'s `evaluate_bpb` function.

---

## Phase 0: Foundation Fixes

No behavior changes. Make code not crash on instantiation.

### bandit/ (10 fixes)

| # | File | Fix |
|---|------|-----|
| 1 | `loop.py` | Fix phantom imports: `bandit.posterior` → `bandit.updater`, `bandit.rollback` → `bandit.safety` |
| 2 | `loop.py` | Fix `dispatch()` arg order: swap `dispatch_ctx` and `state` |
| 3 | `loop.py` | Fix `acceptance.decide()` call: pass `ArmState` and `state` positional args correctly |
| 4 | `loop.py` | Fix `regime_mgr.check_transition()`: add missing `journal_path` arg, unpack tuple return |
| 5 | `loop.py` | Fix `posterior_engine.update()`: match `(state, dispatch_result, log_writer)` signature |
| 6 | `log.py` + `health_audit.py`, `replay_validator.py`, `metrics.py` | Standardize log key on `"type"` everywhere (consumers currently look for `"event"`) |
| 7 | `pipeline.py` | Fix `log_warm_start()`: pass `self.state.to_dict()` not `self.state` |
| 8 | `pipeline.py` | Fix `log_config_change()`: use correct kwarg names (`parameter`, `reason`) |
| 9 | `health.py` | Fix unreachable `elif n > 10`: reorder conditions highest-first |
| 10 | `updater.py` + `loop.py` | Remove double-increment of `global_iteration` |

### gpu_kernels/ (4 fixes)

| # | File | Fix |
|---|------|-----|
| 1 | `pipeline.py` | Fix `KernelConfigManager(config_path=...)` → `KernelConfigManager(config_dir=...)` |
| 2 | `pipeline.py` | Fix `run_generation()`: pass parent dict with `kernel_id`/`kernel_source`/`speedup`, not bare string |
| 3 | `pipeline.py` | Fix `dashboard.render_html(path)`: add `output_path` param to method, write HTML to disk |
| 4 | `pipeline.py` | Wire `self.correctness_verifier` and `self.benchmarker` into discovery/evolution constructors |

### model_scientist/ (4 fixes)

| # | File | Fix |
|---|------|-----|
| 1 | `integration/loop_integrator.py` | Fix `run_at_scales(modification_diff="")`: thread `modified_source` through to `ScaleRunner` |
| 2 | `integration/loop_integrator.py` | Fix empty diagnostics in `_run_metric_evolution`: store and pass diagnostics reports |
| 3 | `metrics/registry.py` | Call `initialize_defaults()` on first load when no file exists |
| 4 | `ablation/diff_parser.py` | Fix `_regions_to_diff()`: generate per-component diffs using the `regions` parameter |

### surrogate_triage/ (7 fixes)

| # | File | Fix |
|---|------|-----|
| 1 | `pipeline.py` | Fix `PaperSourceTracker(path=...)` → `PaperSourceTracker(quality_path=...)` |
| 2 | `pipeline.py` | Fix `ExtractionQualityTracker(data_dir=...)` → `ExtractionQualityTracker(quality_path=...)` |
| 3 | `pipeline.py` | Fix `SurrogateFeedbackLoop(data_dir=...)` → `SurrogateFeedbackLoop(feedback_path=...)` |
| 4 | `pipeline.py` | Fix `NewSourceScout(known_sources_path=...)`: add `__init__` with `known_sources_path` param |
| 5 | `funnel/scoring_pipeline.py` | Fix `surrogate_model.predict()`: unpack `SurrogatePrediction` dataclass fields instead of tuple |
| 6 | `funnel/scoring_pipeline.py` | Fix `enricher.enrich(diff)`: pass `diff.diff_text` and `diff_id=diff.diff_id` |
| 7 | `pipeline.py` | Fix `failure_bridge.feed_rejection()`: add missing `journal_path` arg |

### meta/ (4 fixes)

| # | File | Fix |
|---|------|-----|
| 1 | `experiment/runner.py` | Fix `apply_config()` → `apply()` method name on bridges |
| 2 | `experiment/scheduler.py` | Fix budget check: pass `total_inner_iterations` as first arg, not `total_meta_experiments` twice |
| 3 | `pipeline.py` | Accept sub-layer pipeline refs in constructor, store in `MetaContext` |
| 4 | `baseline/runner.py` | Raise `NotImplementedError` from `_run_single()` instead of returning silent zeros |

### All layers (~30 files)

- Replace `except Exception: pass` with `except Exception as e: logger.exception(e)` in pipeline and loop files (the critical execution paths), using each layer's existing log infrastructure. Leave utility/helper try/except blocks that genuinely need silent fallback.
- Convert all `sys.path.insert(0, ...)` hacks to relative imports

---

## Phase 1: Smoke Tests

Verify Phase 0 worked. No correctness testing — purely "does it boot?"

### Test Structure

```
tests/
  conftest.py              — shared fixtures
  test_imports.py           — all 284 modules import without error
  test_bandit.py            — bandit pipeline instantiation + single iteration
  test_gpu_kernels.py       — gpu_kernels pipeline instantiation
  test_model_scientist.py   — model_scientist pipeline instantiation
  test_surrogate_triage.py  — surrogate_triage pipeline instantiation
  test_meta.py              — meta pipeline instantiation
```

### Shared Fixtures (conftest.py)

- `mock_train_source`: Returns `train.py` contents as a string, read once
- `tmp_data_dir`: Creates a temp directory with minimal required files (empty journal, default state JSON)
- `mock_run_training`: Patches any subprocess/training call to return `{"val_bpb": 0.99, "peak_vram_mb": 4000}`

### Per-Layer Test Pattern

Each test file verifies three things:
1. **Instantiation**: `pipeline = XPipeline(data_dir=tmp_data_dir)` — no crash
2. **Initialize**: `pipeline.initialize()` — no crash
3. **One iteration**: Call the pipeline's main entry point with mocked training — returns a result, not `None` from a swallowed exception

### Import Test

Programmatically discover and import every `.py` file across all 5 layers. Assert no `ImportError`, no `TypeError` from bad relative imports.

---

## Phase 2: Intra-Layer Wiring

Fix important internal bugs within each layer. Add unit tests for key logic.

### bandit/

**Fixes:**
- `sampler.py`: Store the actual selection samples in `SelectionResult`, not a second random draw
- `ceiling_bridge.py`: Apply boost to `diagnostics_boost` field, not `alpha` (preserves evidence conservation invariant)
- Wire `CategoryPromptRouter` into `BanditDispatchRouter.dispatch()` so prompt templates are used
- Make `BanditLoop` use the pipeline's configured component instances via dependency injection instead of lazy imports

**Tests:**
- Thompson sampling selects the arm with the highest sampled value
- Acceptance engine respects per-arm temperature (higher T = more risk tolerance)
- Posterior updates increment the correct arm's alpha/beta

### gpu_kernels/

**Fixes:**
- `extended_divergence.py`, `fallback_verifier.py`: Raise `NotImplementedError` from placeholder methods instead of returning fake zeros
- `integrator.py`: Use actual benchmark result for speedup, or `None` pending benchmark
- `attention_generator.py`: Skip variant 0 (RoPE placeholder) or implement
- `elementwise_generator.py`: Raise error for unrecognized op chains instead of generating identity kernel

**Tests:**
- Elementwise kernel generation for known ops (`relu_square`, `softcap_tanh`, etc.)
- Mutation engine produces syntactically valid AST
- Config manager save/load round-trips correctly

### model_scientist/

**Fixes:**
- `ablation/diff_parser.py`: `_regions_to_diff()` generates per-component diffs using the `regions` parameter
- `journal/writer.py`: Add `update_entry()` method, or change callers to append follow-up entries referencing the original entry ID
- `diagnostics/loss_decomposer.py`: Fix model forward call to match `train.py` signature, or remove unused code path
- `diagnostics/attention_analyzer.py`, `head_clustering.py`: Add model architecture version check so diagnostics fail loudly on architecture mismatch

**Tests:**
- `DiffParser` extracts regions correctly from a sample diff
- `MetricRegistry` initializes 5 default metrics on first load
- `JournalWriter` round-trips entries via write then read

### surrogate_triage/

**Fixes:**
- `schemas.py`: Add `from_dict()` classmethod to `ExtractionQualityRecord`
- `ingestion/paper_filter.py`: Normalize input to always work with `PaperMetadata` objects (add adapter at entry point)
- `intelligence/diagnostics_linker.py`: Wire `BottleneckMapper` to provide actual search terms and boost weights
- `pipeline.py`: Wire `extraction_validator`, `constraint_filter`, `drift_detector` into the pipeline methods that should call them

**Tests:**
- Surrogate trainer `predict()` / `train()` round-trip with synthetic data
- Paper filter scores correctly given keyword matches
- Queue manager deduplicates near-identical entries

### meta/

**Fixes:**
- `convergence/divergence.py`: `DivergenceWatcher.check()` returns mutation instructions rather than mutating state as a side effect
- `knowledge/transfer_validator.py`: `_run_real()` raises `NotImplementedError` instead of returning zeros
- `baseline/improvement_rate.py` + `experiment/runner.py`: Standardize on negative delta = improvement everywhere
- `pipeline.py`: Pass actual default config to `DefaultsVsMetaComparator` instead of `{}`

**Tests:**
- Meta-bandit Thompson sampling selects highest-sampled variant per dimension
- Posterior updater applies three-zone scoring correctly (success/failure/inconclusive)
- Discretizer produces valid variant lists for float, int, and bool parameters

---

## Phase 3: Cross-Layer Integration

Connect the layers to each other. Three integration seams.

### Seam A: Bandit → Sub-Layer Dispatch

**Protocol:**
```python
class DispatchTarget(Protocol):
    def evaluate_modification(self, source: str, context: dict) -> dict: ...
```

model_scientist, surrogate_triage, and gpu_kernels each implement this protocol (or an adapter wraps their existing methods).

**Wiring:**
- `AdaptiveBanditPipeline.__init__` accepts optional `model_scientist`, `surrogate_triage`, `gpu_kernels` pipeline refs
- Populate `LoopContext` with real dispatch targets
- `BanditDispatchRouter` routes by arm category:
  - architecture/optimizer/hyperparameter → `model_scientist.evaluate_modification()`
  - paper-sourced → `surrogate_triage.evaluate_next_paper_candidate()`
  - kernel → `gpu_kernels.run_kernel_discovery_cycle()`

**Test:** Mock all 3 sub-layer pipelines, run one bandit iteration per arm category, verify correct dispatch target was called with expected arguments.

### Seam B: Surrogate Triage → Model Scientist Evaluation

**Wiring:**
- `CandidateRouter`: When a candidate passes the surrogate score threshold, call `model_scientist.evaluate_modification()` with the generated synthetic diff
- Results flow back to `SurrogateRetrainer` as new training data (actual delta vs predicted delta)
- `FailureMiningBridge`: Rejected candidates feed into model_scientist's failure clustering

**Test:** Create a fake high-scoring candidate, verify it reaches model_scientist, verify the evaluation result feeds back to surrogate retrainer.

### Seam C: Meta → All Sub-Layers

**Config reload mechanism:**
Add `reload_overrides(path)` method to model_scientist, surrogate_triage, and gpu_kernels pipelines, matching bandit's existing `HotConfigReloader` pattern. Each checks for `{layer}_overrides.json`, merges into runtime config if file mtime changed.

**Wiring:**
- `MetaAutoresearchPipeline.__init__` accepts 4 sub-layer pipeline instances, stores in `MetaContext`
- `MetaExperimentRunner` calls `bridge.apply(config)` → writes overrides file → sub-layer picks up on next iteration
- Runner calls `pipeline.run_iterations(K)` on actual sub-layer pipelines
- Runner reads `hypothesis_journal.jsonl` to compute improvement rate

**Test:** Set a meta config override for bandit `T_base`, verify: bridge writes `bandit_overrides.json` → bandit reloads → uses new value on next iteration.

---

## Phase 4: Entry Point & E2E

### `run.py` — Orchestration Entry Point

Located at project root. Composes all layers and runs the loop.

```
Arguments:
  --data-dir       Path to data directory (default: ./data)
  --no-meta        Disable meta-optimization layer
  --no-surrogate   Disable surrogate triage (no paper ingestion)
  --no-kernels     Disable GPU kernel generation
  --config         Path to meta_config.json override
  --iterations     Max iterations (default: unlimited)

Minimum viable stack: bandit + model_scientist
Full stack: meta + bandit + model_scientist + surrogate_triage + gpu_kernels
```

Construction order:
1. Instantiate leaf layers (model_scientist, surrogate_triage, gpu_kernels)
2. Instantiate bandit with leaf layer references
3. Instantiate meta with bandit + leaf layer references
4. Initialize all layers
5. Loop: `meta.run_meta_iteration()` (or `bandit.run_iteration()` if `--no-meta`)

Layers are optional: if a layer fails to initialize, log a warning and continue without it.

### Integration Tests (mocked, no GPU needed)

```
tests/test_integration.py:
  test_bandit_dispatches_to_model_scientist()
  test_bandit_dispatches_to_surrogate()
  test_surrogate_feeds_back_to_model_scientist()
  test_meta_config_propagates_to_bandit()
  test_meta_config_propagates_to_all_layers()
  test_full_loop_mocked()  — one complete meta iteration, all training mocked
```

### E2E Test (GPU laptop only)

```
tests/test_e2e.py:
  test_single_real_iteration()
    — runs one actual train.py execution (~5 min)
    — verifies val_bpb result flows through bandit → model_scientist → journal
    — marked with @pytest.mark.e2e, skipped by default
```

Run manually on GPU machine with `pytest -m e2e`.

---

## Out of Scope

- New features or capabilities — this spec is purely about making existing code work
- gpu_kernels ↔ model_scientist integration (kernel metrics feeding into diagnostics) — nice-to-have, not core
- Meta prompt evolution (writing to `bandit/prompt_templates/`) — secondary feature
- Performance optimization — correctness first
- CI/CD pipeline — future work after the code actually runs
