# TODO

Consolidated from the removed superpowers plans/specs. Completed integration work has been omitted; this file keeps only the items that still appear open after auditing the current codebase.

## Remaining integration fixes

- [ ] Restore package/test bootstrap so `pytest` can reliably import repo packages from the repository root. Current audit result: `pytest -q tests/test_integration.py tests/test_e2e.py` fails during collection with `ModuleNotFoundError: No module named 'bandit'`.
- [ ] Add `journal_path` to `LoopContext` in `bandit/schemas.py` and populate it in `AdaptiveBanditPipeline.run_iteration()` in `bandit/pipeline.py`. `bandit/loop.py` already reads `context.journal_path`.
- [ ] Add `run_iterations(n_iterations)` to `AdaptiveBanditPipeline` in `bandit/pipeline.py`. `meta/experiment/runner.py` looks for that method, but the bandit pipeline currently only exposes `run_iteration()`.
- [ ] Define and consistently use `self.journal_path` in `SurrogateTriagePipeline`. `surrogate_triage/pipeline.py` passes `self.journal_path` to `failure_bridge.feed_rejection(...)`, but `__init__()` does not set it.
- [ ] Harden `run.py`: fail loudly when `train.py` is missing instead of silently returning an empty string, and pass the surrogate queue manager into `bandit.run_iteration(...)` so paper-arm dispatch has queue access.

## Test hardening

- [ ] Make `tests/test_integration.py::test_full_loop_mocked` run a meta iteration instead of only checking initialization.
- [ ] Make `tests/test_integration.py::test_meta_config_propagates_to_bandit` assert that the override actually changed bandit state.
- [ ] Strengthen dispatch/integration assertions so the tests verify that the mocked sub-layers were really called.
- [ ] Register the `e2e` pytest marker to remove the current `PytestUnknownMarkWarning`.

## DGM-Hyperagent outer loop

- [ ] Create the `dgm/` package and implement the planned modules: `__init__.py`, `config.py`, `safety.py`, `archive.py`, `selector.py`, `meta_prompt.py`, `cli_agent.py`, `evaluator.py`, `loop.py`, and `__main__.py`.
- [ ] Add tests for DGM safety rules, archive management, parent selection, prompt construction, evaluator behavior, and main-loop orchestration.
- [ ] Verify the CLI entry point works: `python -m dgm.loop --help`.
- [ ] Run the full test suite cleanly after the bootstrap and integration issues above are fixed.
