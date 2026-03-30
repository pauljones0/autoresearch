# DGM-Hyperagent Integration — Design Spec

**Date:** 2026-03-30
**Approach:** Lean reimplementation of DGM-H (facebookresearch/HyperAgents) tailored to AutoResearch, using git worktrees for the archive and Claude Code CLI as the meta agent.

## Context

AutoResearch is Karpathy's autonomous ML research system: an AI agent edits `train.py`, trains for 5 minutes, checks if `val_bpb` improved, keeps or discards, loops forever. Five infrastructure layers (bandit, surrogate_triage, model_scientist, gpu_kernels, meta) were built to upgrade this into a principled research loop.

The HyperAgents paper (Zhang et al., 2026) introduces self-referential agents that can modify not just how they solve tasks but how they improve themselves. The key insight: the meta-level modification procedure is itself editable, enabling metacognitive self-modification.

This spec wraps the AutoResearch infrastructure in a DGM-H outer loop. The infrastructure code (~41K lines across 296 files) becomes the "hyperagent" — the editable program that the meta agent evolves. `train.py` modifications remain the task (inner loop). The DGM-H evolves the researcher, not the research subject.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| What's modifiable | Infrastructure + prompts, not train.py | Clean separation: train.py changes evaluated via val_bpb pipeline |
| LLM transport | Claude Code CLI (`claude -p`) | Free with subscription, no API keys needed |
| Archive model | Git branches + worktrees | Cheap branching, native diffs, full history |
| Evaluation metric | mean(max(-delta, 0)) across K=5 iterations | Captures magnitude and reliability in a single number |
| Meta agent prompt | Guided by eval results + random exploration spotlight | Prevents wild mutations while encouraging breadth |
| Outer loop mutability | Designed modifiable, shipped locked | Safe first runs; flip a flag for full self-reference later |
| Prerequisites | Fix all ~20 integration bugs before first run | Clean baseline, no pre-existing failures |

## Architecture

```
autoresearch/
├── dgm/                          # DGM-H outer loop (PROTECTED)
│   ├── loop.py                   # Main generate loop
│   ├── archive.py                # Git-branch archive manager
│   ├── evaluator.py              # Run K=5 inner iterations, compute score
│   ├── selector.py               # Parent selection (sigmoid + novelty)
│   ├── meta_prompt.py            # Guided meta agent prompt builder
│   ├── cli_agent.py              # Claude Code CLI subprocess wrapper
│   ├── safety.py                 # File-level protection enforcement
│   └── config.py                 # Run configuration
├── bandit/                       # ┐
├── model_scientist/              # │
├── surrogate_triage/             # │ MODIFIABLE by meta agent
├── gpu_kernels/                  # │
├── meta/                         # │
├── run.py                        # ┘
├── train.py                      # PROTECTED — only modified by inner loop
├── prepare.py                    # PROTECTED — ground truth eval
└── tests/                        # PROTECTED
```

## Component Specifications

### 1. Main Loop (`dgm/loop.py`)

Entry point: `python -m dgm.loop --max-generations 28 --k-inner 5`

**Phase 0: Prerequisites**
- Verify all tests pass (`pytest tests/`)
- Verify `train.py` and `prepare.py` exist
- Verify `claude` CLI is available on PATH

**Phase 1: Baseline (Generation 0)**
- Create branch `dgm/gen-0` from master
- Run K=5 inner iterations on unmodified infrastructure
- Record baseline score in `dgm/archive.json`

**Phase 2: Evolution Loop (Generations 1–N)**

For each generation:
1. **SELECT**: Pick parent from archive via `selector.py`
2. **BRANCH**: Create worktree from parent branch (`git worktree add {tempdir}/dgm-gen-N dgm/gen-{parent} -b dgm/gen-N`)
3. **MODIFY**: Invoke Claude Code CLI in worktree via `cli_agent.py` (~10 min timeout)
4. **VALIDATE**: Check no protected files were touched via `safety.py`
5. **EVALUATE**: Run K=5 inner iterations in worktree via `evaluator.py` (~30 min)
6. **ARCHIVE**: Record score, deltas, files_changed, errors in `dgm/archive.json` on master
7. **CLEANUP**: Remove worktree, keep branch

**Resumability:** On restart, reads `archive.json`, finds last completed generation, continues from next.

**Monitoring:** Each generation logs to `dgm/logs/gen-N.log`.

### 2. Archive (`dgm/archive.py`)

**Storage:** `dgm/archive.json` on master branch, committed after each generation.

**Branch naming:** `dgm/gen-{N}` where N is the sequential generation number.

**Schema:**

```json
{
  "generations": {
    "0": {
      "branch": "dgm/gen-0",
      "parent": null,
      "score": 0.0,
      "n_children": 3,
      "n_iterations_completed": 5,
      "mean_negative_delta": 0.0,
      "raw_deltas": [0.01, -0.002, 0.005, 0.0, -0.001],
      "created_at": "2026-03-30T02:00:00Z",
      "meta_agent_prompt_hash": "abc123",
      "files_changed": ["bandit/sampler.py"],
      "commit_sha": "deadbeef",
      "errors": [],
      "eval_duration_seconds": 2940
    }
  },
  "run_config": {
    "k_inner_iterations": 5,
    "max_generations": 28,
    "started_at": "2026-03-30T01:00:00Z"
  }
}
```

**Generation 0** is special: the unmodified master infrastructure, evaluated to establish baseline.

**Key operations:**
- `create_variant(parent_id) -> (gen_id, worktree_path)`: Creates branch and worktree from parent
- `record_score(gen_id, score, deltas, files_changed, errors)`: Updates archive JSON on master
- `get_candidates() -> list[dict]`: Returns all scored generations for parent selection
- `cleanup_worktree(gen_id)`: Removes worktree directory, preserves branch
- `get_best(n=10) -> list[dict]`: Returns top N generations by score

### 3. Evaluator (`dgm/evaluator.py`)

Runs K=5 inner-loop iterations inside a variant's worktree and computes the score.

**Process:**
1. `subprocess.run(["python", "run.py", "--iterations", "5", "--no-meta", "--data-dir", "{worktree}/data"], cwd=worktree_path, timeout=3600)`
2. Parse stdout for val_bpb deltas via regex: `val_bpb:\s+([\d.]+)`
3. Compute delta per iteration: `current_val_bpb - previous_val_bpb`
4. Score = `mean(max(-delta, 0) for delta in all_iterations)`
5. If process crashes, capture traceback; completed iterations score normally, missing iterations count as 0

**Isolation:** Each worktree has its own `data/` directory. No shared state between evaluations. `hypothesis_journal.jsonl`, `strategy_state.json`, etc. are all local to the worktree.

**Timeouts:**
- Full evaluation: 3600 seconds (60 min) hard kill
- Individual inner iteration: inherits from `run.py` (train.py has 300s budget + subprocess 600s timeout)

**Crash handling:** Tracebacks are stored in the archive so the next meta agent can see what broke.

### 4. Parent Selection (`dgm/selector.py`)

Implements the paper's algorithm (Appendix A.2).

**Algorithm:**

For each candidate generation i:

1. Compute dynamic midpoint: `midpoint = mean(top-3 scores in archive)`
2. Sigmoid score: `s_i = 1 / (1 + exp(-10 * (score_i - midpoint)))`
3. Novelty bonus: `h_i = 1 / (1 + n_children_i)`
4. Weight: `w_i = s_i * h_i`
5. Normalize to probability distribution
6. Sample parent from distribution

**Properties:**
- High-scoring variants selected more often
- Variants with many children deprioritized (encourages exploring less-tried parents)
- Nothing is zero-probability (even worst variant can be selected)
- Midpoint adapts as archive improves

**Edge cases:**
- Only gen-0 exists: always select gen-0
- All scores are 0.0: uniform random (sigmoid outputs 0.5 for all)
- Lambda (λ) = 10 (matching paper)

### 5. Meta Agent Prompt (`dgm/meta_prompt.py`)

Builds the guided prompt for Claude Code CLI each iteration.

**Template:**

```
You are a meta agent improving an autonomous ML research system.
Your goal: make the research infrastructure produce better val_bpb
improvements when it runs experiments on train.py.

You may edit, create, or delete any file in:
  bandit/, model_scientist/, surrogate_triage/, gpu_kernels/, meta/, run.py

You must NOT modify:
  prepare.py, train.py, tests/, dgm/

== PARENT VARIANT ==
Branch: dgm/gen-{parent_id}
Score: {parent_score} (mean negative delta across {k} iterations)

== EVALUATION RESULTS (parent's {k} inner iterations) ==
{formatted_deltas_and_errors}

== WHAT THE PARENT CHANGED (from its parent) ==
{parent_diff_summary}

== EXPLORATION SPOTLIGHT ==
Additionally, consider whether improvements could be made in:
  {random_modifiable_file}

== INSTRUCTIONS ==
1. Read the evaluation results above
2. Identify which component caused the worst failures
3. Make a targeted change to improve the system
4. Explain your reasoning in the commit message
```

**Prompt components:**
- **Eval results**: Per-iteration deltas, error messages, tracebacks from parent's run
- **Parent diff**: `git diff {grandparent}..{parent} --stat` showing what the parent already tried
- **Random spotlight**: One randomly selected file from the modifiable set, different each iteration
- **Explicit boundaries**: Allow/deny file lists prevent protected file modification

### 6. CLI Agent Wrapper (`dgm/cli_agent.py`)

Invokes Claude Code as the meta agent via subprocess.

```python
def run_meta_agent(worktree_path, prompt, timeout=600):
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        cwd=worktree_path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return parse_result(result)
```

**Properties:**
- `cwd=worktree_path`: Claude Code's tools see the variant's files, not master
- 10-minute timeout: kills runaway LLM sessions
- Full tool access: Read, Edit, Write, Bash, Grep, Glob inside the worktree
- After return: `git diff --name-only` to detect changes, validate against safety boundaries, commit

**Future upgrade path:** Replace `subprocess.run(["claude", ...])` with `anthropic.Client().messages.create(...)`. Same prompt, same worktree, different transport. Rest of pipeline unchanged.

### 7. Safety Boundaries (`dgm/safety.py`)

**Protected paths:**

| Path | Reason |
|------|--------|
| `prepare.py` | Ground truth evaluation |
| `train.py` | Only modified by inner loop |
| `tests/` | Can't game evaluation |
| `dgm/` | Outer loop (unlock later) |
| `.git/` | Repository internals |
| `pyproject.toml` | Dependencies |
| `uv.lock` | Locked dependencies |

**Enforcement:** After the meta agent returns, `safety.py` checks `git diff --name-only` against the protected list. If any protected file was touched, the variant is rejected (score = 0, error recorded in archive).

**Configuration:**

```python
# dgm/config.py
ALLOW_OUTER_LOOP_MODIFICATION = False
PROTECTED_PATHS = [
    "prepare.py", "train.py", "tests/", "dgm/",
    "pyproject.toml", "uv.lock", ".git/",
]
```

When `ALLOW_OUTER_LOOP_MODIFICATION` is set to True, `dgm/` is removed from the protected list — enabling full self-referential mode.

**Runtime safety:**
- 60-minute hard timeout per evaluation (kill -9)
- Variants run in isolated git worktrees
- Subprocess kill propagates to children
- All subprocess calls logged

### 8. Configuration (`dgm/config.py`)

```python
# Evolution parameters
MAX_GENERATIONS = 28          # ~18 hours overnight
K_INNER_ITERATIONS = 5        # Inner loop iterations per evaluation
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

# Paths
ARCHIVE_PATH = "dgm/archive.json"
LOG_DIR = "dgm/logs"
```

## Prerequisites

Before the first DGM-H run, all integration bugs from the code review must be fixed:

**Tier 1 (Critical — silent no-ops):**
1. Fix `_dispatch_internal` kwargs to match `evaluate_modification()` signature
2. Fix `_dispatch_kernel` to call correct GPU kernel pipeline methods
3. Add `journal_path` field to `LoopContext`
4. Fix meta experiment runner to use bridges (not pipelines) for config application
5. Add `run_iterations(n)` method to `AdaptiveBanditPipeline`

**Tier 2 (High — broken data flows):**
6. Apply paper diffs in `_dispatch_paper` instead of passing base_source unchanged
7. Wire `queue_manager` through `run.py` → `bandit.run_iteration()`
8. Normalize `hypothesis_journal.jsonl` path across all layers
9. Define `self.journal_path` in `SurrogateTriagePipeline.__init__()`
10. Add `import json` to `surrogate_triage/pipeline.py`
11. Add `diff_text` field to `SurrogatePrediction`
12. Add `to_dict()` to `SurrogatePrediction`
13. Fail loudly in `run.py` if `train.py` doesn't exist

**Tier 3 (Medium — degraded functionality):**
14. Store bridges in `MetaContext` or make `_apply_config` use pipeline bridges directly
15. Trigger `pipeline.reload_overrides()` after bridge writes overrides file

**Tier 4 (Test hardening):**
16. Add `mock.assert_called_once_with(...)` to integration tests
17. Make `test_full_loop_mocked` actually call `run_meta_iteration()`
18. Make `test_meta_config_propagates_to_bandit` verify `state.T_base` changed
19. Add true round-trip dispatch test
20. Add config propagation behavior test

## Estimated Timeline

| Phase | What | Duration |
|-------|------|----------|
| Bug fixes | Fix ~20 integration issues from code review | Implementation work |
| DGM-H build | Implement `dgm/` module (~500 lines, 8 files) | Implementation work |
| First run | Generation 0 baseline + 28 evolution generations | ~18 hours overnight |
| Analysis | Inspect archive, compare best variant to baseline | Post-run |

## Out of Scope

- Multi-GPU parallelism (future: evaluate multiple variants simultaneously)
- API key LLM transport (future: swap CLI for direct API calls)
- Full self-reference / outer loop modification (future: flip `ALLOW_OUTER_LOOP_MODIFICATION`)
- Multi-domain evaluation (we use single metric: val_bpb improvement)
- Docker containerization (git worktrees provide sufficient isolation)
- Web dashboard (archive.json is inspectable, analysis scripts can come later)

## References

- [HyperAgents paper (Zhang et al., 2026)](https://arxiv.org/abs/2603.19461)
- [facebookresearch/HyperAgents](https://github.com/facebookresearch/Hyperagents)
- [jennyzzt/dgm (Darwin Gödel Machine)](https://github.com/jennyzzt/dgm)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [awesome-autoresearch](https://github.com/alvinunreal/awesome-autoresearch)
