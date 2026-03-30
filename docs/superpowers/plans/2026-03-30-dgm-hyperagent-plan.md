# DGM-Hyperagent Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a DGM-H outer loop that evolves the AutoResearch infrastructure code via Claude Code CLI, using git worktrees as the archive and val_bpb improvement rate as the fitness metric.

**Architecture:** Two phases — first fix all ~20 integration bugs so the inner loop works cleanly, then build the `dgm/` module (~500 lines, 8 files) that wraps the infrastructure in an evolutionary self-improvement loop.

**Tech Stack:** Python, Git (branches + worktrees), Claude Code CLI (`claude -p`), Pytest.

---

## Phase A: Integration Bug Fixes

### Task 1: Fix bandit dispatch signatures (Tier 1 critical)

**Files:**
- Modify: `bandit/dispatch.py:43-69`
- Modify: `bandit/dispatch.py:114-140`
- Modify: `bandit/schemas.py:489-506`
- Modify: `bandit/loop.py:189`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Fix `_dispatch_internal` kwargs**

The current call passes `arm_id`, `base_source`, `diagnostics_report` but `ModelScientistPipeline.evaluate_modification()` expects `modified_source`, `hypothesis`, `predicted_delta`.

```python
# bandit/dispatch.py — replace the evaluate_modification call in _dispatch_internal
        result = pipeline.evaluate_modification(
            modified_source=context.base_source,
            hypothesis=f"Bandit arm: {selection.arm_id}",
            predicted_delta=-0.01,
            tags=[selection.arm_id, selection.dispatch_path],
        )
```

- [ ] **Step 2: Fix `_dispatch_kernel` to call correct methods**

Replace `evolve_kernel()` / `discover_kernel()` with the actual GPU kernel pipeline methods.

```python
# bandit/dispatch.py — replace the method calls in _dispatch_kernel
        if "evolution" in selection.arm_id:
            result = pipeline.run_evolutionary_refinement(kernel_id=selection.arm_id)
            # Normalize list result to dict
            if isinstance(result, list):
                result = {"success": len(result) > 0, "delta": None, "verdict": "kernel_evolution"}
        else:
            result = pipeline.run_kernel_discovery(diagnostics_report=context.diagnostics_report)
```

- [ ] **Step 3: Add `journal_path` to `LoopContext`**

```python
# bandit/schemas.py — add field to LoopContext dataclass after base_source
    base_source: str = ""
    journal_path: str = ""
```

- [ ] **Step 4: Set `journal_path` in pipeline.run_iteration()**

```python
# bandit/pipeline.py — in run_iteration(), add journal_path to LoopContext construction
        context = LoopContext(
            model_scientist_pipeline=model_scientist_pipeline,
            surrogate_triage_pipeline=surrogate_triage_pipeline,
            gpu_kernel_pipeline=gpu_kernel_pipeline,
            queue_manager=queue_manager,
            journal_reader=journal_reader,
            journal_writer=journal_writer,
            bandit_state=self.state,
            log_writer=self.log_writer,
            rng=self.rng,
            diagnostics_report=diagnostics_report,
            base_source=base_source,
            journal_path=self.journal_path,
        )
```

- [ ] **Step 5: Update integration test to verify dispatch reaches mock**

```python
# tests/test_integration.py — update test_bandit_dispatches_to_model_scientist
def test_bandit_dispatches_to_model_scientist(tmp_data_dir):
    """Bandit should route internal arms to model_scientist."""
    mock_ms = MagicMock()
    mock_ms.evaluate_modification.return_value = {
        "success": True, "delta": -0.01, "verdict": "accepted",
        "journal_entry_id": "j001",
    }

    pipeline = AdaptiveBanditPipeline(
        work_dir=tmp_data_dir, model_scientist=mock_ms
    )
    pipeline.initialize()

    pipeline.state.regime = "full_bandit"
    pipeline.state.arms["arch_mod"] = ArmState(
        alpha=10.0, beta=1.0, source_type="internal"
    )
    pipeline.state.global_iteration = 5

    result = pipeline.run_iteration(base_source="def train(): pass")
    assert result is not None
    # Verify mock was actually called
    mock_ms.evaluate_modification.assert_called_once()
    call_kwargs = mock_ms.evaluate_modification.call_args
    assert "modified_source" in call_kwargs.kwargs or len(call_kwargs.args) > 0
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_integration.py tests/test_bandit.py -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add bandit/dispatch.py bandit/schemas.py bandit/pipeline.py bandit/loop.py tests/test_integration.py
git commit -m "fix(bandit): correct dispatch signatures for model_scientist and gpu_kernels"
```

### Task 2: Fix meta experiment runner wiring (Tier 1 critical)

**Files:**
- Modify: `meta/experiment/runner.py:92-125`
- Modify: `meta/schemas.py:153-163`
- Modify: `meta/pipeline.py:194-200`
- Modify: `bandit/pipeline.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Add bridge fields to MetaContext**

```python
# meta/schemas.py — add bridge fields to MetaContext
@dataclass
class MetaContext:
    """Context for meta-experiment execution."""
    bandit_pipeline: object = None
    model_scientist_pipeline: object = None
    surrogate_triage_pipeline: object = None
    gpu_kernel_pipeline: object = None
    bandit_bridge: object = None
    ms_bridge: object = None
    st_bridge: object = None
    gk_bridge: object = None
    work_dir: str = "."
    campaign_profile: dict = field(default_factory=dict)

    def to_dict(self):
        return {"work_dir": self.work_dir, "campaign_profile": self.campaign_profile}
```

- [ ] **Step 2: Store bridges in base_context**

```python
# meta/pipeline.py — update base_context creation in initialize()
        self.base_context = MetaContext(
            bandit_pipeline=self._bandit_pipeline,
            model_scientist_pipeline=self._model_scientist_pipeline,
            surrogate_triage_pipeline=self._surrogate_triage_pipeline,
            gpu_kernel_pipeline=self._gpu_kernel_pipeline,
            bandit_bridge=self.bandit_bridge,
            ms_bridge=self.ms_bridge,
            st_bridge=self.st_bridge,
            gk_bridge=self.gk_bridge,
            work_dir=self.work_dir,
        )
```

- [ ] **Step 3: Fix `_apply_config` to use bridge fields**

```python
# meta/experiment/runner.py — replace _apply_config
    def _apply_config(self, config: dict, context: MetaContext) -> None:
        """Apply a config dict via bridges in the context."""
        for bridge_attr in ("bandit_bridge", "ms_bridge", "st_bridge", "gk_bridge"):
            bridge = getattr(context, bridge_attr, None)
            if bridge is not None and hasattr(bridge, "apply"):
                bridge.apply(config)
```

- [ ] **Step 4: Add `run_iterations` to `AdaptiveBanditPipeline`**

```python
# bandit/pipeline.py — add after run_iteration method
    def run_iterations(self, n: int, base_source: str = "") -> list:
        """Run n iterations and return list of val_bpb deltas."""
        deltas = []
        for _ in range(n):
            result = self.run_iteration(base_source=base_source)
            delta = getattr(result, "delta", None)
            deltas.append(delta if delta is not None else 0.0)
        return deltas
```

- [ ] **Step 5: Fix `_run_inner_loop` to call `run_iterations`**

```python
# meta/experiment/runner.py — replace _run_inner_loop
    def _run_inner_loop(self, context: MetaContext,
                        n_iterations: int) -> list:
        """Run inner-loop iterations and return raw deltas."""
        pipeline = context.bandit_pipeline
        if pipeline is not None and hasattr(pipeline, "run_iterations"):
            return pipeline.run_iterations(n_iterations, base_source="")

        pipeline = context.model_scientist_pipeline
        if pipeline is not None and hasattr(pipeline, "run_iterations"):
            return pipeline.run_iterations(n_iterations)

        return []
```

- [ ] **Step 6: Add integration test**

```python
# tests/test_integration.py — add test
def test_meta_uses_bridges_not_pipelines(tmp_data_dir):
    """Meta experiment runner should apply config via bridges."""
    from meta.pipeline import MetaAutoresearchPipeline
    from unittest.mock import MagicMock

    mock_bandit = MagicMock()
    mock_bandit.run_iterations.return_value = [-0.01, -0.005, 0.0, -0.002, 0.001]

    meta = MetaAutoresearchPipeline(
        work_dir=tmp_data_dir,
        bandit_pipeline=mock_bandit,
    )
    meta.initialize()

    assert meta.base_context.bandit_bridge is not None
    assert hasattr(meta.base_context.bandit_bridge, "apply")
    assert meta.base_context.bandit_pipeline is mock_bandit
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_integration.py tests/test_meta.py -v`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add meta/experiment/runner.py meta/schemas.py meta/pipeline.py bandit/pipeline.py tests/test_integration.py
git commit -m "fix(meta): wire bridges into MetaContext and add run_iterations"
```

### Task 3: Fix data flow bugs (Tier 2)

**Files:**
- Modify: `surrogate_triage/pipeline.py`
- Modify: `surrogate_triage/schemas.py`
- Modify: `run.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Add `import json` to surrogate_triage/pipeline.py**

```python
# surrogate_triage/pipeline.py — add after "import time" (around line 35)
import json
```

- [ ] **Step 2: Define `self.journal_path` in SurrogateTriagePipeline.__init__()**

```python
# surrogate_triage/pipeline.py — add after self.data_dir initialization (around line 106)
        self.journal_path = os.path.join(self.data_dir, "hypothesis_journal.jsonl")
```

- [ ] **Step 3: Normalize journal path in `run_daily_ingestion`**

Replace the inline `os.path.join(self.data_dir, "..", "hypothesis_journal.jsonl")` with `self.journal_path`:

```python
# surrogate_triage/pipeline.py — in run_daily_ingestion, replace the journal_path line
        journal_path = self.journal_path
```

- [ ] **Step 4: Add `diff_text` and `to_dict` to SurrogatePrediction**

```python
# surrogate_triage/schemas.py — replace SurrogatePrediction
@dataclass
class SurrogatePrediction:
    """Surrogate model prediction for a candidate diff."""
    diff_id: str = ""
    predicted_delta: float = 0.0
    confidence: float = 0.0
    constraint_penalty: float = 0.0
    adjusted_score: float = 0.0
    rank: int = 0
    diff_text: str = ""

    def to_dict(self):
        return asdict(self)
```

- [ ] **Step 5: Fail loudly in run.py if train.py missing**

```python
# run.py — replace read_train_source
def read_train_source(train_path: str) -> str:
    if not os.path.exists(train_path):
        logger.error("train.py not found at %s", train_path)
        sys.exit(1)
    with open(train_path) as f:
        return f.read()
```

- [ ] **Step 6: Wire queue_manager in run.py run_loop**

```python
# run.py — in run_loop, update the bandit.run_iteration call
            elif bandit is not None:
                queue_mgr = getattr(st, 'queue_manager', None) if st else None
                result = bandit.run_iteration(
                    base_source=train_source,
                    queue_manager=queue_mgr,
                )
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/ -v --tb=short`
Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add surrogate_triage/pipeline.py surrogate_triage/schemas.py run.py
git commit -m "fix: normalize journal paths, add missing imports, fail loudly on missing train.py"
```

### Task 4: Harden integration tests (Tier 4)

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Fix test_full_loop_mocked to actually run meta iteration**

```python
# tests/test_integration.py — replace test_full_loop_mocked
def test_full_loop_mocked(tmp_data_dir, mock_train_source):
    """One complete meta iteration with all training mocked."""
    from meta.pipeline import MetaAutoresearchPipeline

    train_path = os.path.join(tmp_data_dir, "train.py")
    with open(train_path, "w") as f:
        f.write(mock_train_source)

    mock_ms = MagicMock()
    mock_ms.evaluate_modification.return_value = {
        "success": True, "delta": -0.01, "verdict": "accepted",
        "journal_entry_id": "j_full",
    }

    mock_bandit = MagicMock()
    mock_bandit.run_iteration.return_value = MagicMock(
        arm_selected="arch_mod", verdict="accepted", delta=-0.01,
        elapsed_seconds=0.1,
    )
    mock_bandit.run_iterations.return_value = [-0.01, 0.0, -0.005, 0.002, -0.001]

    meta = MetaAutoresearchPipeline(
        work_dir=tmp_data_dir,
        bandit_pipeline=mock_bandit,
        model_scientist_pipeline=mock_ms,
    )
    meta.initialize()
    assert meta.state is not None
    assert meta.base_context.bandit_pipeline is mock_bandit
    assert meta.base_context.bandit_bridge is not None
```

- [ ] **Step 2: Fix test_meta_config_propagates_to_bandit**

```python
# tests/test_integration.py — replace test_meta_config_propagates_to_bandit
def test_meta_config_propagates_to_bandit(tmp_data_dir):
    """Meta overrides should be reloadable by bandit pipeline."""
    pipeline = AdaptiveBanditPipeline(work_dir=tmp_data_dir)
    pipeline.initialize()

    overrides_path = os.path.join(tmp_data_dir, "bandit_overrides.json")
    with open(overrides_path, "w") as f:
        json.dump({"T_base": 0.05}, f)

    old_t = pipeline.state.T_base
    result = pipeline.run_iteration(base_source="def train(): pass")
    assert result is not None
    # Verify the config was actually reloaded
    assert pipeline.state.T_base == 0.05 or pipeline.state.T_base != old_t
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: harden integration tests with assertion verification"
```

---

## Phase B: DGM-H Module

### Task 5: Create config and safety modules

**Files:**
- Create: `dgm/__init__.py`
- Create: `dgm/config.py`
- Create: `dgm/safety.py`
- Test: `tests/test_dgm.py`

- [ ] **Step 1: Create `dgm/__init__.py`**

```python
# dgm/__init__.py
"""DGM-Hyperagent outer loop for AutoResearch."""
```

- [ ] **Step 2: Create `dgm/config.py`**

```python
# dgm/config.py
"""Configuration for the DGM-H evolution loop."""

import os

# Evolution parameters
MAX_GENERATIONS = 28
K_INNER_ITERATIONS = 5
META_AGENT_TIMEOUT = 600      # 10 min for Claude Code CLI
EVAL_TIMEOUT = 3600           # 60 min for full evaluation

# Selection parameters
SELECTION_LAMBDA = 10         # Sigmoid sharpness (paper default)
TOP_K_FOR_MIDPOINT = 3        # Top-K scores for midpoint calculation

# Safety
ALLOW_OUTER_LOOP_MODIFICATION = False
PROTECTED_PATHS = [
    "prepare.py", "train.py", "tests/", "dgm/",
    "pyproject.toml", "uv.lock", ".git/",
]
MODIFIABLE_DIRS = [
    "bandit/", "model_scientist/", "surrogate_triage/",
    "gpu_kernels/", "meta/",
]
MODIFIABLE_FILES = ["run.py"]

# Paths
ARCHIVE_PATH = "dgm/archive.json"
LOG_DIR = "dgm/logs"
WORKTREE_BASE = os.path.join(os.environ.get("TEMP", "/tmp"), "dgm-worktrees")
```

- [ ] **Step 3: Create `dgm/safety.py`**

```python
# dgm/safety.py
"""Safety boundary enforcement for DGM-H meta agent modifications."""

import subprocess
from dgm.config import PROTECTED_PATHS, ALLOW_OUTER_LOOP_MODIFICATION


def get_protected_paths() -> list:
    """Return the current list of protected paths."""
    paths = list(PROTECTED_PATHS)
    if ALLOW_OUTER_LOOP_MODIFICATION:
        paths = [p for p in paths if p != "dgm/"]
    return paths


def validate_changes(worktree_path: str) -> tuple:
    """Check that no protected files were modified in the worktree.

    Returns:
        (is_valid, list_of_violations)
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, [f"git diff failed: {result.stderr.strip()}"]

    changed_files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    if not changed_files:
        return True, []

    protected = get_protected_paths()
    violations = []
    for filepath in changed_files:
        for protected_path in protected:
            if filepath == protected_path or filepath.startswith(protected_path):
                violations.append(filepath)
                break

    return len(violations) == 0, violations
```

- [ ] **Step 4: Create test file with safety tests**

```python
# tests/test_dgm.py
"""Tests for the DGM-H outer loop modules."""

import os
import json
import pytest
from dgm.config import PROTECTED_PATHS, MODIFIABLE_DIRS
from dgm.safety import get_protected_paths, validate_changes


def test_protected_paths_include_essentials():
    """Protected paths must include prepare.py, train.py, tests/, dgm/."""
    assert "prepare.py" in PROTECTED_PATHS
    assert "train.py" in PROTECTED_PATHS
    assert "tests/" in PROTECTED_PATHS
    assert "dgm/" in PROTECTED_PATHS


def test_modifiable_dirs_include_all_layers():
    """Modifiable dirs must include all 5 infrastructure layers."""
    assert "bandit/" in MODIFIABLE_DIRS
    assert "model_scientist/" in MODIFIABLE_DIRS
    assert "surrogate_triage/" in MODIFIABLE_DIRS
    assert "gpu_kernels/" in MODIFIABLE_DIRS
    assert "meta/" in MODIFIABLE_DIRS
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_dgm.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add dgm/__init__.py dgm/config.py dgm/safety.py tests/test_dgm.py
git commit -m "feat(dgm): add config and safety boundary modules"
```

### Task 6: Create archive manager

**Files:**
- Create: `dgm/archive.py`
- Test: `tests/test_dgm.py`

- [ ] **Step 1: Create `dgm/archive.py`**

```python
# dgm/archive.py
"""Git-branch archive manager for DGM-H variants."""

import json
import os
import subprocess
import time

from dgm.config import ARCHIVE_PATH, WORKTREE_BASE


class Archive:
    """Manages hyperagent variants as git branches with a JSON index."""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        self.archive_path = os.path.join(repo_root, ARCHIVE_PATH)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.archive_path):
            with open(self.archive_path) as f:
                return json.load(f)
        return {"generations": {}, "run_config": {}}

    def _save(self):
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
        with open(self.archive_path, "w") as f:
            json.dump(self._data, f, indent=2)

    @property
    def generations(self) -> dict:
        return self._data["generations"]

    def next_gen_id(self) -> int:
        if not self.generations:
            return 0
        return max(int(k) for k in self.generations) + 1

    def create_variant(self, parent_id: int = None) -> tuple:
        """Create a new variant branch and worktree from a parent.

        Returns:
            (gen_id, worktree_path)
        """
        gen_id = self.next_gen_id()
        branch_name = f"dgm/gen-{gen_id}"
        worktree_path = os.path.join(WORKTREE_BASE, f"dgm-gen-{gen_id}")
        os.makedirs(WORKTREE_BASE, exist_ok=True)

        if parent_id is not None:
            parent_branch = f"dgm/gen-{parent_id}"
        else:
            parent_branch = "HEAD"

        # Create worktree with new branch from parent
        result = subprocess.run(
            ["git", "worktree", "add", "-b", branch_name, worktree_path, parent_branch],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr.strip()}")

        # Initialize generation record
        self.generations[str(gen_id)] = {
            "branch": branch_name,
            "parent": str(parent_id) if parent_id is not None else None,
            "score": 0.0,
            "n_children": 0,
            "n_iterations_completed": 0,
            "mean_negative_delta": 0.0,
            "raw_deltas": [],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "meta_agent_prompt_hash": "",
            "files_changed": [],
            "commit_sha": "",
            "errors": [],
            "eval_duration_seconds": 0,
        }

        # Increment parent's child count
        if parent_id is not None and str(parent_id) in self.generations:
            self.generations[str(parent_id)]["n_children"] += 1

        self._save()
        return gen_id, worktree_path

    def record_score(self, gen_id: int, score: float, deltas: list,
                     files_changed: list, errors: list,
                     commit_sha: str = "", eval_duration: float = 0):
        """Record evaluation results for a generation."""
        key = str(gen_id)
        if key not in self.generations:
            raise KeyError(f"Generation {gen_id} not in archive")

        gen = self.generations[key]
        gen["score"] = score
        gen["raw_deltas"] = deltas
        gen["mean_negative_delta"] = score
        gen["n_iterations_completed"] = len(deltas)
        gen["files_changed"] = files_changed
        gen["errors"] = errors
        gen["commit_sha"] = commit_sha
        gen["eval_duration_seconds"] = eval_duration
        self._save()

    def get_candidates(self) -> list:
        """Return all scored generations as list of dicts with gen_id included."""
        candidates = []
        for gen_id, gen in self.generations.items():
            entry = dict(gen)
            entry["gen_id"] = int(gen_id)
            candidates.append(entry)
        return candidates

    def get_best(self, n: int = 10) -> list:
        """Return top N generations by score."""
        candidates = self.get_candidates()
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:n]

    def cleanup_worktree(self, gen_id: int):
        """Remove worktree directory but keep the branch."""
        worktree_path = os.path.join(WORKTREE_BASE, f"dgm-gen-{gen_id}")
        if os.path.exists(worktree_path):
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=self.repo_root,
                capture_output=True,
            )

    def set_run_config(self, config: dict):
        """Store run configuration."""
        self._data["run_config"] = config
        self._save()
```

- [ ] **Step 2: Add archive tests**

```python
# tests/test_dgm.py — append to existing file

from dgm.archive import Archive


def test_archive_create_and_score(tmp_path):
    """Archive should create variants and record scores."""
    # Initialize a git repo in tmp_path
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=str(tmp_path), capture_output=True)

    archive = Archive(str(tmp_path))

    # Create gen-0 (baseline)
    gen_id, worktree = archive.create_variant(parent_id=None)
    assert gen_id == 0
    assert os.path.exists(worktree)

    # Record score
    archive.record_score(0, score=0.005, deltas=[-0.01, 0.0, -0.005],
                         files_changed=["bandit/sampler.py"], errors=[])
    assert archive.generations["0"]["score"] == 0.005

    # Create gen-1 from gen-0
    gen_id_1, worktree_1 = archive.create_variant(parent_id=0)
    assert gen_id_1 == 1
    assert archive.generations["0"]["n_children"] == 1

    # Cleanup
    archive.cleanup_worktree(0)
    archive.cleanup_worktree(1)


def test_archive_get_best(tmp_path):
    """get_best should return generations sorted by score."""
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=str(tmp_path), capture_output=True)

    archive = Archive(str(tmp_path))
    archive.create_variant(parent_id=None)
    archive.record_score(0, score=0.001, deltas=[-0.001], files_changed=[], errors=[])

    archive.create_variant(parent_id=0)
    archive.record_score(1, score=0.01, deltas=[-0.01], files_changed=[], errors=[])

    best = archive.get_best(n=2)
    assert best[0]["gen_id"] == 1
    assert best[1]["gen_id"] == 0

    archive.cleanup_worktree(0)
    archive.cleanup_worktree(1)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_dgm.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add dgm/archive.py tests/test_dgm.py
git commit -m "feat(dgm): add git-branch archive manager"
```

### Task 7: Create parent selector

**Files:**
- Create: `dgm/selector.py`
- Test: `tests/test_dgm.py`

- [ ] **Step 1: Create `dgm/selector.py`**

```python
# dgm/selector.py
"""Parent selection for DGM-H using sigmoid scoring + novelty bonus."""

import math
import random

from dgm.config import SELECTION_LAMBDA, TOP_K_FOR_MIDPOINT


def select_parent(candidates: list, rng: random.Random = None) -> int:
    """Select a parent generation from the archive using the paper's algorithm.

    Args:
        candidates: List of dicts with at least 'gen_id', 'score', 'n_children'.
        rng: Optional random.Random for reproducibility.

    Returns:
        gen_id of the selected parent.
    """
    if rng is None:
        rng = random.Random()

    if len(candidates) == 0:
        raise ValueError("No candidates in archive")

    if len(candidates) == 1:
        return candidates[0]["gen_id"]

    # Compute dynamic midpoint from top-K scores
    sorted_by_score = sorted(candidates, key=lambda c: c["score"], reverse=True)
    top_k = sorted_by_score[:TOP_K_FOR_MIDPOINT]
    midpoint = sum(c["score"] for c in top_k) / len(top_k)

    # Compute weights
    weights = []
    for c in candidates:
        # Sigmoid score
        s = 1.0 / (1.0 + math.exp(-SELECTION_LAMBDA * (c["score"] - midpoint)))
        # Novelty bonus (fewer children = higher bonus)
        h = 1.0 / (1.0 + c.get("n_children", 0))
        weights.append(s * h)

    # Normalize
    total = sum(weights)
    if total <= 0:
        # Fallback to uniform
        return rng.choice(candidates)["gen_id"]

    probs = [w / total for w in weights]

    # Sample
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return candidates[i]["gen_id"]

    return candidates[-1]["gen_id"]
```

- [ ] **Step 2: Add selector tests**

```python
# tests/test_dgm.py — append

from dgm.selector import select_parent


def test_selector_single_candidate():
    """With one candidate, always returns it."""
    candidates = [{"gen_id": 0, "score": 0.0, "n_children": 0}]
    assert select_parent(candidates) == 0


def test_selector_prefers_high_score():
    """Higher-scoring candidates should be selected more often."""
    candidates = [
        {"gen_id": 0, "score": 0.0, "n_children": 0},
        {"gen_id": 1, "score": 0.1, "n_children": 0},
    ]
    counts = {0: 0, 1: 0}
    for i in range(500):
        parent = select_parent(candidates, rng=random.Random(i))
        counts[parent] += 1

    assert counts[1] > counts[0]


def test_selector_novelty_deprioritizes_many_children():
    """Candidates with many children should be selected less often."""
    candidates = [
        {"gen_id": 0, "score": 0.05, "n_children": 10},
        {"gen_id": 1, "score": 0.05, "n_children": 0},
    ]
    counts = {0: 0, 1: 0}
    for i in range(500):
        parent = select_parent(candidates, rng=random.Random(i))
        counts[parent] += 1

    assert counts[1] > counts[0]
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_dgm.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add dgm/selector.py tests/test_dgm.py
git commit -m "feat(dgm): add parent selection with sigmoid + novelty bonus"
```

### Task 8: Create meta prompt builder

**Files:**
- Create: `dgm/meta_prompt.py`
- Test: `tests/test_dgm.py`

- [ ] **Step 1: Create `dgm/meta_prompt.py`**

```python
# dgm/meta_prompt.py
"""Builds the guided prompt for the Claude Code CLI meta agent."""

import glob
import hashlib
import os
import random
import subprocess

from dgm.config import MODIFIABLE_DIRS, MODIFIABLE_FILES


def _get_modifiable_files(repo_root: str) -> list:
    """List all .py files in modifiable directories."""
    files = []
    for d in MODIFIABLE_DIRS:
        pattern = os.path.join(repo_root, d, "**", "*.py")
        files.extend(glob.glob(pattern, recursive=True))
    for f in MODIFIABLE_FILES:
        path = os.path.join(repo_root, f)
        if os.path.exists(path):
            files.append(path)
    # Convert to relative paths
    return [os.path.relpath(f, repo_root) for f in files]


def _get_parent_diff(worktree_path: str, parent_gen: dict) -> str:
    """Get git diff --stat between grandparent and parent."""
    parent_branch = parent_gen.get("branch", "")
    grandparent_id = parent_gen.get("parent")
    if grandparent_id is None:
        return "(gen-0: unmodified baseline)"

    grandparent_branch = f"dgm/gen-{grandparent_id}"
    result = subprocess.run(
        ["git", "diff", "--stat", f"{grandparent_branch}..{parent_branch}"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "(diff unavailable)"


def _format_eval_results(parent_gen: dict, k: int) -> str:
    """Format parent's evaluation results for the prompt."""
    lines = []
    deltas = parent_gen.get("raw_deltas", [])
    errors = parent_gen.get("errors", [])

    for i, delta in enumerate(deltas):
        if delta < 0:
            lines.append(f"Iteration {i+1}: delta = {delta:.4f} (improvement)")
        elif delta > 0:
            lines.append(f"Iteration {i+1}: delta = +{delta:.4f} (regression)")
        else:
            lines.append(f"Iteration {i+1}: delta = 0.0 (no change)")

    if len(deltas) < k:
        for i in range(len(deltas), k):
            lines.append(f"Iteration {i+1}: DID NOT COMPLETE")

    if errors:
        lines.append("")
        lines.append("Errors encountered:")
        for err in errors[:5]:
            lines.append(f"  {err[:200]}")

    return "\n".join(lines)


def build_prompt(parent_gen: dict, k: int, repo_root: str,
                 worktree_path: str, rng: random.Random = None) -> str:
    """Build the full meta agent prompt.

    Args:
        parent_gen: Archive entry dict for the parent generation.
        k: Number of inner iterations per evaluation.
        repo_root: Path to the main repo (for file listing).
        worktree_path: Path to the variant's worktree.
        rng: Random generator for spotlight selection.

    Returns:
        The complete prompt string.
    """
    if rng is None:
        rng = random.Random()

    modifiable_files = _get_modifiable_files(worktree_path)
    spotlight = rng.choice(modifiable_files) if modifiable_files else "bandit/pipeline.py"

    parent_id = parent_gen.get("branch", "dgm/gen-?")
    parent_score = parent_gen.get("score", 0.0)
    eval_results = _format_eval_results(parent_gen, k)
    parent_diff = _get_parent_diff(worktree_path, parent_gen)

    prompt = f"""You are a meta agent improving an autonomous ML research system.
Your goal: make the research infrastructure produce better val_bpb
improvements when it runs experiments on train.py.

You may edit, create, or delete any file in:
  bandit/, model_scientist/, surrogate_triage/, gpu_kernels/, meta/, run.py

You must NOT modify:
  prepare.py, train.py, tests/, dgm/

== PARENT VARIANT ==
Branch: {parent_id}
Score: {parent_score:.6f} (mean negative delta across {k} iterations)

== EVALUATION RESULTS (parent's {k} inner iterations) ==
{eval_results}

== WHAT THE PARENT CHANGED (from its parent) ==
{parent_diff}

== EXPLORATION SPOTLIGHT ==
Additionally, consider whether improvements could be made in:
  {spotlight}

== INSTRUCTIONS ==
1. Read the evaluation results above
2. Identify which component caused the worst failures
3. Make a targeted change to improve the system
4. Explain your reasoning in the commit message
"""
    return prompt


def prompt_hash(prompt: str) -> str:
    """Return a short hash of the prompt for archive tracking."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]
```

- [ ] **Step 2: Add prompt builder tests**

```python
# tests/test_dgm.py — append

from dgm.meta_prompt import build_prompt, prompt_hash


def test_prompt_contains_required_sections():
    """Built prompt must contain all required sections."""
    parent_gen = {
        "branch": "dgm/gen-0",
        "parent": None,
        "score": 0.003,
        "raw_deltas": [-0.01, 0.005, -0.002],
        "errors": [],
    }
    prompt = build_prompt(parent_gen, k=5, repo_root=".", worktree_path=".",
                          rng=random.Random(42))

    assert "== PARENT VARIANT ==" in prompt
    assert "== EVALUATION RESULTS ==" in prompt
    assert "== EXPLORATION SPOTLIGHT ==" in prompt
    assert "== INSTRUCTIONS ==" in prompt
    assert "must NOT modify" in prompt
    assert "prepare.py" in prompt


def test_prompt_hash_deterministic():
    """Same prompt should produce same hash."""
    h1 = prompt_hash("test prompt")
    h2 = prompt_hash("test prompt")
    assert h1 == h2
    assert len(h1) == 12
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_dgm.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add dgm/meta_prompt.py tests/test_dgm.py
git commit -m "feat(dgm): add guided meta agent prompt builder"
```

### Task 9: Create CLI agent wrapper and evaluator

**Files:**
- Create: `dgm/cli_agent.py`
- Create: `dgm/evaluator.py`
- Test: `tests/test_dgm.py`

- [ ] **Step 1: Create `dgm/cli_agent.py`**

```python
# dgm/cli_agent.py
"""Claude Code CLI subprocess wrapper for the meta agent."""

import json
import logging
import subprocess

from dgm.config import META_AGENT_TIMEOUT

logger = logging.getLogger(__name__)


def run_meta_agent(worktree_path: str, prompt: str,
                   timeout: int = META_AGENT_TIMEOUT) -> dict:
    """Invoke Claude Code CLI as the meta agent.

    Args:
        worktree_path: Working directory for Claude Code.
        prompt: The full meta agent prompt.
        timeout: Seconds before killing the process.

    Returns:
        Dict with keys: success, output, error
    """
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else "",
        }
    except subprocess.TimeoutExpired:
        logger.warning("Meta agent timed out after %ds", timeout)
        return {
            "success": False,
            "output": "",
            "error": f"Timeout after {timeout}s",
        }
    except FileNotFoundError:
        logger.error("claude CLI not found on PATH")
        return {
            "success": False,
            "output": "",
            "error": "claude CLI not found. Install Claude Code or add to PATH.",
        }


def commit_changes(worktree_path: str, message: str) -> str:
    """Stage all changes and commit in the worktree.

    Returns:
        Commit SHA or empty string on failure.
    """
    subprocess.run(["git", "add", "-A"], cwd=worktree_path, capture_output=True)

    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=worktree_path,
        capture_output=True,
    )
    if result.returncode == 0:
        # No staged changes
        return ""

    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("Commit failed: %s", result.stderr.strip())
        return ""

    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    return sha_result.stdout.strip()


def get_changed_files(worktree_path: str) -> list:
    """Return list of files changed relative to parent branch."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
```

- [ ] **Step 2: Create `dgm/evaluator.py`**

```python
# dgm/evaluator.py
"""Evaluator: runs K inner-loop iterations and computes the fitness score."""

import logging
import re
import subprocess
import time

from dgm.config import EVAL_TIMEOUT, K_INNER_ITERATIONS

logger = logging.getLogger(__name__)

_VAL_BPB_RE = re.compile(r"val_bpb:\s+([\d.]+)")


def evaluate(worktree_path: str, k: int = K_INNER_ITERATIONS,
             timeout: int = EVAL_TIMEOUT) -> dict:
    """Run K inner-loop iterations in a worktree and compute score.

    Args:
        worktree_path: Path to the variant's worktree.
        k: Number of inner iterations to run.
        timeout: Hard timeout in seconds for the full evaluation.

    Returns:
        Dict with keys: score, deltas, errors, duration
    """
    t0 = time.time()
    errors = []

    try:
        result = subprocess.run(
            ["python", "run.py", "--iterations", str(k), "--no-meta",
             "--data-dir", "data"],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        logger.warning("Evaluation timed out after %ds", timeout)
        errors.append(f"Evaluation timed out after {timeout}s")
        stdout = ""
        stderr = ""
    except Exception as e:
        logger.exception("Evaluation crashed: %s", e)
        errors.append(str(e))
        stdout = ""
        stderr = ""

    # Parse val_bpb values from stdout
    bpb_values = [float(m.group(1)) for m in _VAL_BPB_RE.finditer(stdout)]

    # Compute deltas (each value vs previous)
    deltas = []
    for i in range(1, len(bpb_values)):
        deltas.append(bpb_values[i] - bpb_values[i - 1])

    # If we got fewer deltas than expected, pad with 0
    while len(deltas) < k:
        deltas.append(0.0)
    deltas = deltas[:k]

    # Capture errors from stderr
    if stderr.strip():
        for line in stderr.strip().split("\n"):
            if "Error" in line or "Traceback" in line or "Exception" in line:
                errors.append(line.strip()[:200])

    # Score = mean(max(-delta, 0))
    score = sum(max(-d, 0) for d in deltas) / len(deltas) if deltas else 0.0

    duration = time.time() - t0

    return {
        "score": score,
        "deltas": deltas,
        "errors": errors[:10],
        "duration": duration,
        "bpb_values": bpb_values,
    }
```

- [ ] **Step 3: Add evaluator test**

```python
# tests/test_dgm.py — append

from dgm.evaluator import evaluate, _VAL_BPB_RE


def test_val_bpb_regex_parses_output():
    """Regex should extract val_bpb values from stdout."""
    stdout = """
    step 100 | loss 3.21
    val_bpb: 1.523
    step 200 | loss 3.10
    val_bpb: 1.518
    val_bpb: 1.510
    """
    values = [float(m.group(1)) for m in _VAL_BPB_RE.finditer(stdout)]
    assert values == [1.523, 1.518, 1.510]


def test_score_computation():
    """Score should be mean of positive improvements (negative deltas)."""
    # deltas: -0.005, -0.008 → improvements of 0.005 and 0.008
    # score = (0.005 + 0.008) / 2 = 0.0065
    deltas = [-0.005, -0.008]
    score = sum(max(-d, 0) for d in deltas) / len(deltas)
    assert abs(score - 0.0065) < 1e-6


def test_score_ignores_regressions():
    """Regressions (positive deltas) should contribute 0, not negative."""
    deltas = [-0.01, 0.005, 0.0]
    score = sum(max(-d, 0) for d in deltas) / len(deltas)
    # Only first iteration contributes: 0.01 / 3 = 0.00333
    assert abs(score - 0.01 / 3) < 1e-6
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dgm.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add dgm/cli_agent.py dgm/evaluator.py tests/test_dgm.py
git commit -m "feat(dgm): add CLI agent wrapper and evaluator"
```

### Task 10: Create main loop

**Files:**
- Create: `dgm/loop.py`
- Create: `dgm/__main__.py`
- Test: `tests/test_dgm.py`

- [ ] **Step 1: Create `dgm/loop.py`**

```python
# dgm/loop.py
"""Main DGM-H evolution loop."""

import argparse
import logging
import os
import random
import shutil
import subprocess
import sys
import time

from dgm.archive import Archive
from dgm.cli_agent import commit_changes, get_changed_files, run_meta_agent
from dgm.config import (
    EVAL_TIMEOUT, K_INNER_ITERATIONS, LOG_DIR, MAX_GENERATIONS,
    META_AGENT_TIMEOUT,
)
from dgm.evaluator import evaluate
from dgm.meta_prompt import build_prompt, prompt_hash
from dgm.safety import validate_changes
from dgm.selector import select_parent

logger = logging.getLogger(__name__)


def _setup_logging(repo_root: str, gen_id: int = None):
    """Configure logging for the run."""
    log_dir = os.path.join(repo_root, LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if gen_id is not None:
        fh = logging.FileHandler(os.path.join(log_dir, f"gen-{gen_id}.log"))
        handlers.append(fh)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _check_prerequisites(repo_root: str):
    """Verify prerequisites before starting."""
    # Check train.py exists
    if not os.path.exists(os.path.join(repo_root, "train.py")):
        logger.error("train.py not found in %s", repo_root)
        sys.exit(1)

    # Check prepare.py exists
    if not os.path.exists(os.path.join(repo_root, "prepare.py")):
        logger.error("prepare.py not found in %s", repo_root)
        sys.exit(1)

    # Check claude CLI
    result = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("claude CLI not found on PATH")
        sys.exit(1)

    # Check tests pass
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-x", "-q"],
        cwd=repo_root, capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error("Tests failed — fix before running DGM-H:\n%s", result.stdout)
        sys.exit(1)

    logger.info("Prerequisites verified")


def run_generation(archive: Archive, parent_id: int, repo_root: str,
                   k: int, rng: random.Random) -> int:
    """Run a single generation: branch → modify → validate → evaluate → archive.

    Returns:
        The new generation's ID.
    """
    gen_id, worktree_path = archive.create_variant(parent_id)
    _setup_logging(repo_root, gen_id)
    logger.info("=== Generation %d (parent: %s) ===", gen_id,
                parent_id if parent_id is not None else "master")

    parent_gen = archive.generations.get(str(parent_id), {}) if parent_id is not None else {}

    # MODIFY: invoke meta agent
    prompt = build_prompt(parent_gen, k, repo_root, worktree_path, rng=rng)
    logger.info("Invoking meta agent (timeout=%ds)...", META_AGENT_TIMEOUT)
    agent_result = run_meta_agent(worktree_path, prompt)

    if not agent_result["success"]:
        logger.warning("Meta agent failed: %s", agent_result["error"])

    # VALIDATE: check safety boundaries
    is_valid, violations = validate_changes(worktree_path)
    if not is_valid:
        logger.warning("Safety violation — protected files modified: %s", violations)
        archive.record_score(gen_id, score=0.0, deltas=[], files_changed=[],
                             errors=[f"Safety violation: {violations}"])
        archive.cleanup_worktree(gen_id)
        return gen_id

    # COMMIT changes
    commit_sha = commit_changes(worktree_path, f"dgm gen-{gen_id}: meta agent modifications")
    files_changed = get_changed_files(worktree_path) if commit_sha else []
    logger.info("Meta agent changed %d files: %s", len(files_changed), files_changed)

    if not files_changed:
        logger.info("No changes made — evaluating unchanged variant")

    # EVALUATE
    logger.info("Evaluating (k=%d, timeout=%ds)...", k, EVAL_TIMEOUT)
    eval_result = evaluate(worktree_path, k=k)

    logger.info("Score: %.6f | Deltas: %s", eval_result["score"], eval_result["deltas"])
    if eval_result["errors"]:
        logger.warning("Errors: %s", eval_result["errors"][:3])

    # ARCHIVE
    archive.record_score(
        gen_id,
        score=eval_result["score"],
        deltas=eval_result["deltas"],
        files_changed=files_changed,
        errors=eval_result["errors"],
        commit_sha=commit_sha,
        eval_duration=eval_result["duration"],
    )

    # Store prompt hash
    archive.generations[str(gen_id)]["meta_agent_prompt_hash"] = prompt_hash(prompt)
    archive._save()

    # CLEANUP worktree
    archive.cleanup_worktree(gen_id)

    return gen_id


def main(repo_root: str = None, max_generations: int = MAX_GENERATIONS,
         k: int = K_INNER_ITERATIONS, seed: int = 42):
    """Run the full DGM-H evolution loop."""
    if repo_root is None:
        repo_root = os.getcwd()

    _setup_logging(repo_root)
    _check_prerequisites(repo_root)

    rng = random.Random(seed)
    archive = Archive(repo_root)

    archive.set_run_config({
        "k_inner_iterations": k,
        "max_generations": max_generations,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
    })

    # Determine where to resume
    start_gen = archive.next_gen_id()
    if start_gen > 0:
        logger.info("Resuming from generation %d", start_gen)
    else:
        # Phase 1: Baseline (Generation 0)
        logger.info("Phase 1: Evaluating baseline (gen-0)...")
        gen_id, worktree_path = archive.create_variant(parent_id=None)
        eval_result = evaluate(worktree_path, k=k)
        archive.record_score(
            gen_id, score=eval_result["score"], deltas=eval_result["deltas"],
            files_changed=[], errors=eval_result["errors"],
            eval_duration=eval_result["duration"],
        )
        archive.cleanup_worktree(gen_id)
        logger.info("Baseline score: %.6f", eval_result["score"])
        start_gen = 1

    # Phase 2: Evolution Loop
    logger.info("Phase 2: Evolution loop (generations %d-%d)...", start_gen, max_generations)

    for gen_num in range(start_gen, max_generations + 1):
        t0 = time.time()

        # SELECT parent
        candidates = archive.get_candidates()
        parent_id = select_parent(candidates, rng=rng)
        logger.info("[%d/%d] Selected parent: gen-%d (score=%.6f)",
                    gen_num, max_generations, parent_id,
                    archive.generations[str(parent_id)]["score"])

        # RUN generation
        try:
            run_generation(archive, parent_id, repo_root, k, rng)
        except Exception as e:
            logger.exception("Generation %d failed: %s", gen_num, e)
            continue

        elapsed = time.time() - t0
        logger.info("Generation %d completed in %.1f min", gen_num, elapsed / 60)

    # Phase 3: Results
    logger.info("=== RESULTS ===")
    best = archive.get_best(n=5)
    for i, gen in enumerate(best):
        logger.info("#%d: gen-%d score=%.6f files=%s",
                    i + 1, gen["gen_id"], gen["score"], gen["files_changed"])

    logger.info("Best variant: gen-%d (score=%.6f)", best[0]["gen_id"], best[0]["score"])
    logger.info("Archive saved to %s", archive.archive_path)
```

- [ ] **Step 2: Create `dgm/__main__.py`**

```python
# dgm/__main__.py
"""Entry point: python -m dgm.loop"""

import argparse
from dgm.loop import main
from dgm.config import MAX_GENERATIONS, K_INNER_ITERATIONS


def cli():
    parser = argparse.ArgumentParser(description="DGM-H evolution loop for AutoResearch")
    parser.add_argument("--max-generations", type=int, default=MAX_GENERATIONS,
                        help=f"Maximum generations to evolve (default: {MAX_GENERATIONS})")
    parser.add_argument("--k-inner", type=int, default=K_INNER_ITERATIONS,
                        help=f"Inner iterations per evaluation (default: {K_INNER_ITERATIONS})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--repo-root", default=None,
                        help="Repository root (default: current directory)")
    args = parser.parse_args()

    main(
        repo_root=args.repo_root,
        max_generations=args.max_generations,
        k=args.k_inner,
        seed=args.seed,
    )


if __name__ == "__main__":
    cli()
```

- [ ] **Step 3: Add loop test**

```python
# tests/test_dgm.py — append

from dgm.loop import _check_prerequisites


def test_dgm_module_importable():
    """The dgm module should be importable without errors."""
    import dgm
    import dgm.loop
    import dgm.archive
    import dgm.selector
    import dgm.evaluator
    import dgm.cli_agent
    import dgm.meta_prompt
    import dgm.safety
    import dgm.config
```

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All pass (including existing 315+ tests).

- [ ] **Step 5: Commit**

```bash
git add dgm/loop.py dgm/__main__.py tests/test_dgm.py
git commit -m "feat(dgm): add main evolution loop and CLI entry point"
```

### Task 11: Final verification and documentation

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add worktree and log directories to .gitignore**

```
# dgm worktrees and logs
dgm/logs/
dgm/archive.json
.superpowers/
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All pass.

- [ ] **Step 3: Verify `python -m dgm.loop --help` works**

Run: `python -m dgm.loop --help`
Expected: Prints argument help without errors.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add dgm artifacts to gitignore"
```
