# AutoResearch Multi-Layer Integration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 27 critical bugs and integrate 5 layers of the AutoResearch project (bandit, surrogate_triage, model_scientist, gpu_kernels, meta) to create a functional, end-to-end autonomous research loop.

**Architecture:** Bottom-up integration. We first fix method signatures and imports so the code can instantiate (Phase 0), verify with smoke tests (Phase 1), fix internal logic and add unit tests (Phase 2), wire cross-layer dependencies (Phase 3), and finally build the orchestration entry point (Phase 4).

**Tech Stack:** Python, Pytest, JSONL (journaling), Triton (GPU kernels), Thompson Sampling (bandit).

---

## Phase 0: Foundation Fixes

### Task 1: Bandit Foundation Fixes (Part 1: imports and signatures)

**Files:**
- Modify: `bandit/loop.py`
- Modify: `bandit/pipeline.py`

- [ ] **Step 1: Fix imports in `bandit/loop.py`**
Replace phantom imports: `bandit.posterior` -> `bandit.updater`, `bandit.rollback` -> `bandit.safety`.
```python
# bandit/loop.py
from bandit.updater import PosteriorEngine # was bandit.posterior
from bandit.safety import RollbackManager  # was bandit.rollback
```

- [ ] **Step 2: Fix `dispatch()` argument order in `bandit/loop.py`**
Swap `dispatch_ctx` and `state`.
```python
# bandit/loop.py
def dispatch(self, state: BanditState, dispatch_ctx: dict): # Ensure this order matches callers
```

- [ ] **Step 3: Fix `acceptance.decide()` call in `bandit/loop.py`**
Pass `ArmState` and `state` positional args correctly.
```python
# bandit/loop.py
decision = self.acceptance.decide(arm_state, state) # Ensure correct positional args
```

- [ ] **Step 4: Fix `regime_mgr.check_transition()` call in `bandit/loop.py`**
Add missing `journal_path` arg and unpack tuple return.
```python
# bandit/loop.py
new_regime, transitioned = self.regime_mgr.check_transition(state, self.journal_path)
```

- [ ] **Step 5: Fix `posterior_engine.update()` signature in `bandit/loop.py`**
Match `(state, dispatch_result, log_writer)`.
```python
# bandit/loop.py
self.posterior_engine.update(state, dispatch_result, self.log_writer)
```

- [ ] **Step 6: Commit**
```bash
git add bandit/loop.py bandit/pipeline.py
git commit -m "fix(bandit): correct signatures and imports in loop.py"
```

### Task 2: Bandit Foundation Fixes (Part 2: logging and state)

**Files:**
- Modify: `bandit/log.py`
- Modify: `bandit/health_audit.py`
- Modify: `bandit/validation/replay_validator.py`
- Modify: `bandit/metrics.py`
- Modify: `bandit/pipeline.py`
- Modify: `bandit/health.py`
- Modify: `bandit/updater.py`

- [ ] **Step 1: Standardize log key on "type"**
Search and replace `"event"` with `"type"` in log records across `bandit/log.py`, `bandit/health_audit.py`, `bandit/validation/replay_validator.py`, and `bandit/metrics.py`.

- [ ] **Step 2: Fix `log_warm_start()` in `bandit/pipeline.py`**
Pass `self.state.to_dict()` instead of `self.state`.
```python
# bandit/pipeline.py
self.logger.log_warm_start(self.state.to_dict())
```

- [ ] **Step 3: Fix `log_config_change()` in `bandit/pipeline.py`**
Use correct kwarg names (`parameter`, `reason`).
```python
# bandit/pipeline.py
self.logger.log_config_change(parameter=param, reason=reason)
```

- [ ] **Step 4: Fix `health.py` logic ordering**
Reorder `elif n > 10` to be checked before smaller values if they overlap.
```python
# bandit/health.py
if n > 100: ...
elif n > 10: ... # Ensure highest-first
```

- [ ] **Step 5: Remove double-increment of `global_iteration`**
Check `bandit/updater.py` and `bandit/loop.py` to ensure `state.global_iteration += 1` happens exactly once per loop.

- [ ] **Step 6: Commit**
```bash
git add bandit/*.py bandit/validation/*.py
git commit -m "fix(bandit): standardize logging and fix state increments"
```

### Task 3: GPU Kernels Foundation Fixes

**Files:**
- Modify: `gpu_kernels/pipeline.py`

- [ ] **Step 1: Fix `KernelConfigManager` instantiation**
Change `config_path=...` to `config_dir=...`.
```python
# gpu_kernels/pipeline.py
self.config_manager = KernelConfigManager(config_dir=self.config_dir)
```

- [ ] **Step 2: Fix `run_generation()` return/passing**
Pass parent dict with `kernel_id`/`kernel_source`/`speedup`, not bare string.

- [ ] **Step 3: Fix `dashboard.render_html()` signature**
Add `output_path` param to method and ensure it writes HTML to disk.

- [ ] **Step 4: Wire verifier and benchmarker**
Pass `self.correctness_verifier` and `self.benchmarker` into discovery/evolution constructors in `gpu_kernels/pipeline.py`.

- [ ] **Step 5: Commit**
```bash
git add gpu_kernels/pipeline.py
git commit -m "fix(gpu_kernels): correct config manager and wiring"
```

### Task 4: Model Scientist Foundation Fixes

**Files:**
- Modify: `model_scientist/integration/loop_integrator.py`
- Modify: `model_scientist/metrics/registry.py`
- Modify: `model_scientist/ablation/diff_parser.py`

- [ ] **Step 1: Fix `run_at_scales` signature**
Thread `modified_source` through to `ScaleRunner`.
```python
# model_scientist/integration/loop_integrator.py
def run_at_scales(self, modification_diff: str, modified_source: str):
```

- [ ] **Step 2: Fix diagnostics in `_run_metric_evolution`**
Store and pass diagnostics reports instead of passing empty objects.

- [ ] **Step 3: Fix `MetricRegistry` initialization**
Call `initialize_defaults()` on first load when no file exists in `model_scientist/metrics/registry.py`.

- [ ] **Step 4: Fix `_regions_to_diff()` in `model_scientist/ablation/diff_parser.py`**
Generate per-component diffs using the `regions` parameter.

- [ ] **Step 5: Commit**
```bash
git add model_scientist/integration/loop_integrator.py model_scientist/metrics/registry.py model_scientist/ablation/diff_parser.py
git commit -m "fix(model_scientist): thread source through scales and fix registry init"
```

### Task 5: Surrogate Triage Foundation Fixes

**Files:**
- Modify: `surrogate_triage/pipeline.py`
- Modify: `surrogate_triage/funnel/scoring_pipeline.py`

- [ ] **Step 1: Fix tracker paths in `surrogate_triage/pipeline.py`**
Update `PaperSourceTracker`, `ExtractionQualityTracker`, and `SurrogateFeedbackLoop` to use `quality_path` or `feedback_path` instead of `path` or `data_dir`.

- [ ] **Step 2: Fix `NewSourceScout` instantiation**
Add `__init__` with `known_sources_path` param if missing.

- [ ] **Step 3: Fix `surrogate_model.predict()` unpacking**
Unpack `SurrogatePrediction` dataclass fields instead of treating as tuple in `surrogate_triage/funnel/scoring_pipeline.py`.

- [ ] **Step 4: Fix `enricher.enrich(diff)` call**
Pass `diff.diff_text` and `diff_id=diff.diff_id`.

- [ ] **Step 5: Fix `failure_bridge.feed_rejection()`**
Add missing `journal_path` arg.

- [ ] **Step 6: Commit**
```bash
git add surrogate_triage/pipeline.py surrogate_triage/funnel/scoring_pipeline.py
git commit -m "fix(surrogate_triage): fix paths and dataclass unpacking"
```

### Task 6: Meta Foundation Fixes

**Files:**
- Modify: `meta/experiment/runner.py`
- Modify: `meta/experiment/scheduler.py`
- Modify: `meta/pipeline.py`
- Modify: `meta/baseline/runner.py`

- [ ] **Step 1: Fix `apply_config()` method name**
Rename `apply_config()` calls to `apply()` on bridges in `meta/experiment/runner.py`.

- [ ] **Step 2: Fix budget check in `meta/experiment/scheduler.py`**
Pass `total_inner_iterations` as first arg, not `total_meta_experiments` twice.

- [ ] **Step 3: Fix `MetaAutoresearchPipeline` constructor**
Accept sub-layer pipeline refs and store in `MetaContext`.

- [ ] **Step 4: Fix `baseline/runner.py` silent zeros**
Raise `NotImplementedError` from `_run_single()` instead of returning zeros.

- [ ] **Step 5: Commit**
```bash
git add meta/experiment/runner.py meta/experiment/scheduler.py meta/pipeline.py meta/baseline/runner.py
git commit -m "fix(meta): correct method names and budget checks"
```

### Task 7: Global Foundation Hygiene

**Files:**
- Modify: ~30 files across all layers (pipeline and loop files)

- [ ] **Step 1: Standardize exception logging**
Replace `except Exception: pass` with `except Exception as e: logger.exception(e)` in all pipeline and loop files.

- [ ] **Step 2: Convert to relative imports**
Replace all `sys.path.insert(0, ...)` hacks with proper relative imports (e.g., `from ..bandit import ...`).

- [ ] **Step 3: Commit**
```bash
git commit -am "chore: standardize exception logging and relative imports"
```

---

## Phase 1: Smoke Tests

### Task 8: Test Infrastructure and Import Test

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_imports.py`

- [ ] **Step 1: Create `tests/conftest.py` with shared fixtures**
Implement `mock_train_source`, `tmp_data_dir`, and `mock_run_training` (patching subprocess calls to return `{"val_bpb": 0.99, "peak_vram_mb": 4000}`).

- [ ] **Step 2: Create `tests/test_imports.py`**
Programmatically discover and import every `.py` file across all 5 layers.
```python
import pkgutil
import importlib
import pytest

@pytest.mark.parametrize("module_info", pkgutil.walk_packages(["bandit", "gpu_kernels", "model_scientist", "surrogate_triage", "meta"]))
def test_import_all(module_info):
    importlib.import_module(module_info.name)
```

- [ ] **Step 3: Run import tests**
Run: `pytest tests/test_imports.py`
Expected: 0 errors.

- [ ] **Step 4: Commit**
```bash
git add tests/conftest.py tests/test_imports.py
git commit -m "test: add import smoke test and conftest fixtures"
```

### Task 9: Per-Layer Smoke Tests

**Files:**
- Create: `tests/test_bandit.py`
- Create: `tests/test_gpu_kernels.py`
- Create: `tests/test_model_scientist.py`
- Create: `tests/test_surrogate_triage.py`
- Create: `tests/test_meta.py`

- [ ] **Step 1: Create `tests/test_bandit.py`**
Test instantiation, initialization, and one iteration of the bandit pipeline.

- [ ] **Step 2: Create `tests/test_gpu_kernels.py`**
Test instantiation of the gpu_kernels pipeline.

- [ ] **Step 3: Create `tests/test_model_scientist.py`**
Test instantiation of the model_scientist pipeline.

- [ ] **Step 4: Create `tests/test_surrogate_triage.py`**
Test instantiation of the surrogate_triage pipeline.

- [ ] **Step 5: Create `tests/test_meta.py`**
Test instantiation of the meta pipeline.

- [ ] **Step 6: Run per-layer smoke tests**
Run: `pytest tests/test_bandit.py tests/test_gpu_kernels.py ...`
Expected: All pass.

- [ ] **Step 7: Commit**
```bash
git add tests/test_*.py
git commit -m "test: add per-layer smoke tests"
```

---

## Phase 2: Intra-Layer Wiring

### Task 10: Bandit Internal Wiring Fixes

**Files:**
- Modify: `bandit/sampler.py`
- Modify: `bandit/ceiling_bridge.py`
- Modify: `bandit/dispatch.py`
- Modify: `bandit/loop.py`

- [ ] **Step 1: Fix `sampler.py`**
Store the actual selection samples in `SelectionResult` instead of a second random draw.

- [ ] **Step 2: Fix `ceiling_bridge.py`**
Apply boost to `diagnostics_boost` field, not `alpha`.

- [ ] **Step 3: Wire `CategoryPromptRouter`**
In `BanditDispatchRouter.dispatch()`, ensure prompt templates are utilized.

- [ ] **Step 4: Dependency Injection in `BanditLoop`**
Use pipeline's configured component instances instead of lazy imports.

- [ ] **Step 5: Add unit tests in `tests/test_bandit.py`**
Verify Thompson sampling selection logic and acceptance engine risk tolerance (higher T = more risk).

- [ ] **Step 6: Commit**
```bash
git add bandit/*.py tests/test_bandit.py
git commit -m "fix(bandit): correct sampler logic and prompt routing"
```

### Task 11: GPU Kernels Internal Wiring Fixes

**Files:**
- Modify: `gpu_kernels/extended_divergence.py`
- Modify: `gpu_kernels/fallback_verifier.py`
- Modify: `gpu_kernels/integrator.py`
- Modify: `gpu_kernels/attention_generator.py`
- Modify: `gpu_kernels/elementwise_generator.py`

- [ ] **Step 1: Remove fake zeros**
In `extended_divergence.py` and `fallback_verifier.py`, raise `NotImplementedError` from placeholders.

- [ ] **Step 2: Fix `integrator.py` speedup calculation**
Use actual benchmark result for speedup, or `None` if pending.

- [ ] **Step 3: Fix `attention_generator.py`**
Skip variant 0 (RoPE placeholder) or implement it.

- [ ] **Step 4: Fix `elementwise_generator.py`**
Raise error for unrecognized op chains.

- [ ] **Step 5: Add unit tests in `tests/test_gpu_kernels.py`**
Verify elementwise kernel generation and config manager round-trip.

- [ ] **Step 6: Commit**
```bash
git add gpu_kernels/*.py tests/test_gpu_kernels.py
git commit -m "fix(gpu_kernels): replace fake returns with NotImplementedError"
```

### Task 12: Model Scientist Internal Wiring Fixes

**Files:**
- Modify: `model_scientist/ablation/diff_parser.py`
- Modify: `model_scientist/journal/writer.py`
- Modify: `model_scientist/diagnostics/loss_decomposer.py`
- Modify: `model_scientist/diagnostics/attention_analyzer.py`

- [ ] **Step 1: Fix `diff_parser.py`**
Generate per-component diffs using the `regions` parameter.

- [ ] **Step 2: Fix `journal/writer.py`**
Add `update_entry()` method or change callers to append follow-up entries referencing original IDs.

- [ ] **Step 3: Fix `loss_decomposer.py` signature**
Fix model forward call to match `train.py`.

- [ ] **Step 4: Add architecture checks**
In `attention_analyzer.py` and `head_clustering.py`, add model version check.

- [ ] **Step 5: Add unit tests in `tests/test_model_scientist.py`**
Verify `DiffParser` region extraction and `MetricRegistry` initialization.

- [ ] **Step 6: Commit**
```bash
git add model_scientist/**/*.py tests/test_model_scientist.py
git commit -m "fix(model_scientist): fix diff parser and journal writer"
```

### Task 13: Surrogate Triage Internal Wiring Fixes

**Files:**
- Modify: `surrogate_triage/schemas.py`
- Modify: `surrogate_triage/ingestion/paper_filter.py`
- Modify: `surrogate_triage/intelligence/diagnostics_linker.py`
- Modify: `surrogate_triage/pipeline.py`

- [ ] **Step 1: Add `from_dict()` to `ExtractionQualityRecord`**
In `surrogate_triage/schemas.py`.

- [ ] **Step 2: Normalize `paper_filter.py` inputs**
Always work with `PaperMetadata` objects.

- [ ] **Step 3: Wire `BottleneckMapper`**
Provide actual search terms and boost weights in `diagnostics_linker.py`.

- [ ] **Step 4: Wire pipeline components**
Connect `extraction_validator`, `constraint_filter`, and `drift_detector` in `surrogate_triage/pipeline.py`.

- [ ] **Step 5: Add unit tests in `tests/test_surrogate_triage.py`**
Verify surrogate trainer predict/train round-trip and paper filter keyword scoring.

- [ ] **Step 6: Commit**
```bash
git add surrogate_triage/**/*.py tests/test_surrogate_triage.py
git commit -m "fix(surrogate_triage): wire missing pipeline components"
```

### Task 14: Meta Internal Wiring Fixes

**Files:**
- Modify: `meta/convergence/divergence.py`
- Modify: `meta/knowledge/transfer_validator.py`
- Modify: `meta/baseline/improvement_rate.py`
- Modify: `meta/pipeline.py`

- [ ] **Step 1: Fix `DivergenceWatcher.check()`**
Return mutation instructions instead of mutating state as a side effect.

- [ ] **Step 2: Fix `transfer_validator.py`**
Raise `NotImplementedError` from `_run_real()` instead of returning zeros.

- [ ] **Step 3: Standardize improvement delta**
Ensure negative delta = improvement everywhere in `baseline/improvement_rate.py` and `experiment/runner.py`.

- [ ] **Step 4: Fix `DefaultsVsMetaComparator`**
Pass actual default config instead of `{}` in `meta/pipeline.py`.

- [ ] **Step 5: Add unit tests in `tests/test_meta.py`**
Verify meta-bandit Thompson sampling and three-zone scoring logic.

- [ ] **Step 6: Commit**
```bash
git add meta/**/*.py tests/test_meta.py
git commit -m "fix(meta): standardize improvement metrics and fix side effects"
```

---

## Phase 3: Cross-Layer Integration

### Task 15: Seam A: Bandit to Sub-Layer Dispatch

**Files:**
- Modify: `bandit/pipeline.py`
- Modify: `bandit/dispatch.py`

- [ ] **Step 1: Update `AdaptiveBanditPipeline.__init__`**
Accept optional `model_scientist`, `surrogate_triage`, and `gpu_kernels` pipeline refs.

- [ ] **Step 2: Populate `LoopContext`**
Store real dispatch targets in `LoopContext`.

- [ ] **Step 3: Implement Routing in `BanditDispatchRouter`**
Route arm categories (architecture/optimizer/hyperparameter, paper-sourced, kernel) to the corresponding sub-layer methods.

- [ ] **Step 4: Verify with integration test**
Add `test_bandit_dispatches_to_model_scientist` in `tests/test_integration.py`.

- [ ] **Step 5: Commit**
```bash
git add bandit/pipeline.py bandit/dispatch.py
git commit -m "feat(integration): wire bandit to sub-layer dispatch"
```

### Task 16: Seam B: Surrogate Triage to Model Scientist

**Files:**
- Modify: `surrogate_triage/pipeline.py`

- [ ] **Step 1: Wire `CandidateRouter` to `model_scientist`**
When a candidate passes surrogate threshold, call `model_scientist.evaluate_modification()`.

- [ ] **Step 2: Wire feedback loop**
Feed results back to `SurrogateRetrainer` as new training data.

- [ ] **Step 3: Verify with integration test**
Add `test_surrogate_feeds_back_to_model_scientist` in `tests/test_integration.py`.

- [ ] **Step 4: Commit**
```bash
git add surrogate_triage/pipeline.py
git commit -m "feat(integration): wire surrogate triage to model scientist"
```

### Task 17: Seam C: Meta to All Sub-Layers

**Files:**
- Modify: `model_scientist/pipeline.py`
- Modify: `surrogate_triage/pipeline.py`
- Modify: `gpu_kernels/pipeline.py`
- Modify: `meta/pipeline.py`

- [ ] **Step 1: Add `reload_overrides(path)` to sub-layer pipelines**
Implement standard hot-reloading pattern using `mtime` check.

- [ ] **Step 2: Wire `MetaAutoresearchPipeline`**
Accept 4 sub-layer pipeline instances in `__init__` and store in `MetaContext`.

- [ ] **Step 3: Update `MetaExperimentRunner`**
Call `bridge.apply(config)` to write overrides file, then run sub-layer iterations.

- [ ] **Step 4: Verify with integration test**
Add `test_meta_config_propagates_to_bandit` in `tests/test_integration.py`.

- [ ] **Step 5: Commit**
```bash
git add meta/pipeline.py */pipeline.py
git commit -m "feat(integration): wire meta to all sub-layers via hot-reload"
```

---

## Phase 4: Entry Point & E2E

### Task 18: Orchestration Entry Point (run.py)

**Files:**
- Create: `run.py`

- [ ] **Step 1: Implement `run.py` orchestration**
Handle CLI arguments (`--data-dir`, `--no-meta`, etc.), instantiate layers in order, initialize them, and start the loop.

- [ ] **Step 2: Add integration tests in `tests/test_integration.py`**
Implement `test_full_loop_mocked()` to verify one complete meta iteration.

- [ ] **Step 3: Create E2E test in `tests/test_e2e.py`**
Implement `test_single_real_iteration()` marked with `@pytest.mark.e2e`.

- [ ] **Step 4: Verify entry point**
Run: `python run.py --iterations 1 --no-meta` (mocked training)
Expected: Successful completion of one iteration.

- [ ] **Step 5: Commit**
```bash
git add run.py tests/test_integration.py tests/test_e2e.py
git commit -m "feat: add orchestration entry point and full integration tests"
```
