# Model Scientist Pipeline — Multi-Phase Implementation Plan

**Source:** Synergy C from `ideas_v2.md` (Risk 5/7, Reward 7/7)
**Status:** NOT IMPLEMENTED
**Goal:** Build a full scientific method loop — diagnose, hypothesize, experiment, ablate, document, and learn from failure — so the system becomes a better scientist with each iteration.

---

## Phase 1: Diagnostics & Documentation Foundation

**Gate:** Phase 1 is complete when the diagnostic reporter produces a structured JSON report after every training run AND the hypothesis journal persists at least 10 entries with predictions vs. outcomes. Manual review required before proceeding to Phase 2.

### Subphase 1.1: Gradient & Activation Diagnostics

**Status:** NOT IMPLEMENTED
**Gate:** Passes when a test training run produces a valid `diagnostics_report.json` with all required fields and no NaN values.

| Agent | Task |
|-------|------|
| **DiagnosticsInstrumenter** | Instrument `train.py` to capture per-layer gradient norms, activation means/stds, and dead neuron counts at configurable intervals during training. Store raw tensors in memory, summarize to JSON at end of run. |
| **AttentionAnalyzer** | Add attention entropy computation per head and attention pattern collapse detection (near-uniform distributions). Integrate into the diagnostics report under an `attention` key. |
| **LossDecomposer** | Implement loss decomposition by token frequency bucket (top-1k, 1k-10k, 10k+, rare). Add to diagnostics report under a `loss_decomposition` key. |
| **DiagnosticsValidator** | Write a validation script that loads `diagnostics_report.json`, checks schema completeness, value ranges, and flags anomalies. Run against 3 short training runs to confirm stability. |

### Subphase 1.2: Interpretability Probes

**Status:** NOT IMPLEMENTED
**Gate:** Passes when probing classifiers achieve >60% accuracy on at least 2 linguistic tasks (e.g., POS tagging, dependency depth) using frozen model representations, confirming they extract meaningful signal.

| Agent | Task |
|-------|------|
| **ProbeTrainer** | Implement lightweight linear probing classifiers that train on frozen intermediate representations to detect what information is encoded at each layer (syntax, semantics, position). |
| **CKASimilarityAgent** | Implement Centered Kernel Alignment (CKA) between layer pairs to identify redundant computation. Add a `layer_similarity_matrix` to the diagnostics report. |
| **HeadClusterer** | Cluster attention heads by pattern type (positional, syntactic, rare-token-focused) using attention weight statistics. Add `head_clusters` to diagnostics report. |

### Subphase 1.3: Hypothesis Journal

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the journal system can record, retrieve, and query entries, and a manual review of 5+ entries confirms the schema captures diagnosis → hypothesis → prediction → outcome faithfully.

| Agent | Task |
|-------|------|
| **JournalArchitect** | Design and implement `hypothesis_journal.jsonl` — an append-only structured log. Each entry: `{id, timestamp, diagnostics_summary, hypothesis, predicted_delta, actual_delta, modification_diff, verdict, tags}`. |
| **JournalWriter** | Integrate journal writing into the autoresearch loop so every experiment (accepted or rejected) is logged with full context. |
| **JournalReader** | Build a query interface: filter by tag, date range, verdict, predicted vs. actual delta. Expose as a CLI tool and as context the research agent can read. |

---

## Phase 2: Failure Mining & Scaling Analysis

**Gate:** Phase 2 is complete when the failure pattern database contains at least 20 clustered failure patterns AND the scaling predictor correctly forecasts (within 15% relative error) the direction of 3 consecutive held-out modifications. Manual review of failure clusters required.

### Subphase 2.1: Failure Pattern Mining

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the miner clusters at least 10 rejected experiments into 3+ distinct failure patterns, and the constraint generator produces human-readable negative constraints.

| Agent | Task |
|-------|------|
| **FailureExtractor** | Parse `hypothesis_journal.jsonl` for rejected experiments. Extract structured features: modification category, model state at time of attempt (diagnostics snapshot), predicted vs. actual delta, failure mode (regression, instability, no change). |
| **FailureClusterer** | Cluster extracted failures by similarity (modification type + model state). Use simple distance metrics on feature vectors — no ML model needed initially. Output: `failure_patterns.json` with pattern descriptions and instance counts. |
| **ConstraintGenerator** | Convert failure clusters into natural-language negative constraints (e.g., "Attention head pruning when gradient norms in layers 1-3 are below 0.01 has failed 7/8 times"). Inject these constraints into the research agent's prompt context. |
| **ConstraintValidator** | Back-test constraints against historical journal entries to measure precision/recall. A constraint is valid if it would have filtered >50% of failures in its category without filtering >10% of successes. |

### Subphase 2.2: Multi-Scale Testing Infrastructure

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the scaling harness can train at 3 different scales (0.25x, 0.5x, 1x parameter count) with consistent config derivation, and produces a scaling curve JSON for a test modification.

| Agent | Task |
|-------|------|
| **ScaleConfigDeriver** | Build a scale config generator: given the base `train.py` config, derive consistent configs at 0.25x and 0.5x parameter count by adjusting depth, width, and heads proportionally. Ensure architectural validity at each scale. |
| **ScaleRunner** | Implement a multi-scale training harness that runs a modification at all scales sequentially (smallest first, early-exit if smallest fails). Collect val_bpb at each scale. |
| **ScalingCurveFitter** | Fit power-law scaling curves to multi-scale results. Predict improvement at 1x scale from smaller-scale data. Output: `scaling_prediction.json` with extrapolated delta and confidence interval. |

### Subphase 2.3: Scale-Gated Acceptance

**Status:** NOT IMPLEMENTED
**Gate:** Passes when 5 consecutive modifications are processed through the scale gate, and at least 1 modification is correctly rejected (improvement vanishes at scale) that would have been accepted without scaling analysis.

| Agent | Task |
|-------|------|
| **ScaleGateIntegrator** | Wire the scaling predictor into the acceptance pipeline. Before a modification proceeds to full evaluation, it must pass the scaling gate: predicted improvement at 1x must exceed a configurable threshold (default: 50% of small-scale improvement retained). |
| **ScaleGateLogger** | Log all scale-gate decisions to the hypothesis journal with full scaling curve data, enabling future analysis of the gate's accuracy. |

---

## Phase 3: Causal Ablation Engine

**Gate:** Phase 3 is complete when the ablation engine successfully decomposes at least 5 multi-component modifications into individual marginal contributions, and at least 1 modification has a component stripped that was neutral or negative. Manual review of ablation reports required.

### Subphase 3.1: Modification Decomposition

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the decomposer correctly splits 3 test modifications (manually created) into independent components, and each component applies cleanly in isolation.

| Agent | Task |
|-------|------|
| **DiffParser** | Build a code diff parser that takes a modification to `train.py` and identifies semantically independent components (e.g., "activation function change" vs. "initialization change" vs. "width adjustment"). Use AST-level analysis where possible, fall back to heuristic line-group splitting. |
| **ComponentIsolator** | For each identified component, generate an isolated patch that applies only that component. Verify each isolated patch applies cleanly and produces a runnable `train.py`. |

### Subphase 3.2: Ablation Runner

**Status:** NOT IMPLEMENTED
**Gate:** Passes when a full ablation of a 3-component modification completes (full mod, 3 leave-one-out runs, baseline) and produces a marginal contribution table.

| Agent | Task |
|-------|------|
| **AblationOrchestrator** | Implement leave-one-out ablation: for an N-component modification, run N+1 training evaluations (full modification + N versions each missing one component). Compute marginal contribution of each component as `full_improvement - leave_one_out_improvement`. |
| **AblationBudgeter** | Add compute budget awareness: skip ablation for single-component modifications, limit ablation to top-K most impactful modifications per cycle, and set a max ablation compute budget as a percentage of total training compute (default: 15%). |

### Subphase 3.3: Component Stripping

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the stripper correctly removes a neutral/negative component from a test modification and the stripped version achieves equal or better val_bpb than the full modification.

| Agent | Task |
|-------|------|
| **ComponentStripper** | After ablation, automatically strip components with marginal contribution <= 0 (neutral or harmful). Re-evaluate the stripped modification to confirm improvement is preserved or enhanced. |
| **AblationJournalWriter** | Extend the hypothesis journal to record ablation details: component list, marginal contributions, stripped components, final vs. original val_bpb. Feed stripped-component patterns into the failure mining pipeline. |

---

## Phase 4: Dynamic Metrics & Critic Evolution

**Gate:** Phase 4 is complete when the metric evolution system has proposed at least 10 new metrics, promoted at least 3 that correlate with modification success (r > 0.3), pruned at least 2 that don't, and the critic produces metric proposals that differ meaningfully from the initial hardcoded set. Full manual review of promoted metrics and critic behavior required before declaring pipeline complete.

### Subphase 4.1: Critic Agent — Metric Proposal

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the critic proposes 5+ novel metrics given a real diagnostics report, and at least 3 are computable (implementation succeeds without errors) and produce non-trivial variance across training runs.

| Agent | Task |
|-------|------|
| **CriticPromptDesigner** | Design the critic agent's prompt: given a diagnostics report and the current metric inventory, propose 1-3 new diagnostic metrics that would better capture the model's current bottleneck. The critic must output: metric name, computation method, and rationale. |
| **MetricImplementer** | Take critic-proposed metric definitions and generate executable Python code. Run the code against stored diagnostics data to verify it produces valid numerical output. Add passing metrics to the live diagnostics pipeline. |
| **MetricRegistry** | Build `metric_registry.json` — a catalog of all active, proposed, and retired metrics with metadata: source (hardcoded vs. critic-proposed), creation date, correlation with modification success, current status (active/candidate/retired). |

### Subphase 4.2: Metric Evolution — Selection Pressure

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the evolution engine has run at least 3 selection cycles, and the set of active metrics has changed (at least 1 promotion and 1 retirement) based on predictive power.

| Agent | Task |
|-------|------|
| **MetricCorrelator** | After every N experiments (configurable, default 10), compute the correlation between each metric's pre-experiment value and the experiment's outcome (accepted/rejected, delta magnitude). Rank metrics by predictive power. |
| **MetricPromoter** | Promote candidate metrics with correlation > threshold (default r > 0.3) to active status (included in the research agent's context). Retire active metrics that fall below the threshold for 2 consecutive evaluation cycles. |
| **ContextBudgetManager** | Manage the research agent's diagnostic context window budget. As metrics are promoted/retired, rebalance what the agent sees. High-predictive-power metrics get more context space; low-value metrics are summarized or omitted. |

### Subphase 4.3: Closing the Loop

**Status:** NOT IMPLEMENTED
**Gate:** Passes when an end-to-end cycle completes: diagnostics → critic proposes metric → metric is implemented → metric is evaluated over 10+ experiments → metric is promoted or retired based on predictive power. This confirms the full feedback loop is operational.

| Agent | Task |
|-------|------|
| **LoopIntegrator** | Wire all Phase 4 components into the main autoresearch loop. Ensure the critic runs during GPU-bound training downtime, metric implementation is sandboxed, and the evolution engine runs on schedule. |
| **SafetyGuard** | Implement guardrails: max number of active metrics (default 20), max critic proposals per cycle (default 3), mandatory human review for metrics that touch the training loop (vs. post-hoc analysis only). Prevent metric proliferation from degrading system performance. |
| **PipelineMonitor** | Build a monitoring dashboard (CLI or simple HTML) showing: active metrics and their correlations, recent critic proposals, promotion/retirement history, and pipeline health (latency, errors, resource usage). |

---

## Full Pipeline Integration

**Gate:** The Model Scientist Pipeline is **IMPLEMENTED** when all four phases pass their gates AND an end-to-end demonstration shows the following sequence completing without human intervention:

1. Training run completes → diagnostics report generated (Phase 1)
2. Research agent reads diagnostics + evolved metrics + failure constraints → proposes modification
3. Modification tested at multiple scales → passes scale gate (Phase 2)
4. Accepted modification ablated → neutral components stripped (Phase 3)
5. Full record written to hypothesis journal (Phase 1)
6. Failure patterns updated (Phase 2)
7. Metric evolution cycle runs (Phase 4)

| Agent | Task |
|-------|------|
| **EndToEndValidator** | Run the full pipeline for 20 consecutive iterations. Verify all 7 steps execute, all logs are written, and no stage silently fails. |
| **RegressionTester** | Confirm the pipeline produces equal or better val_bpb than the baseline autoresearch loop over a 50-iteration A/B comparison. |
| **DocumentationAgent** | Update README.md and create an operator guide covering: configuration, monitoring, common failure modes, and how to interpret the hypothesis journal and metric registry. |

---

## Agent Summary

| Agent Name | Phase | Subphase | Role |
|---|---|---|---|
| DiagnosticsInstrumenter | 1 | 1.1 | Instrument train.py for gradient/activation capture |
| AttentionAnalyzer | 1 | 1.1 | Attention entropy & collapse detection |
| LossDecomposer | 1 | 1.1 | Per-bucket loss decomposition |
| DiagnosticsValidator | 1 | 1.1 | Schema validation & anomaly detection |
| ProbeTrainer | 1 | 1.2 | Linear probing classifiers on frozen reps |
| CKASimilarityAgent | 1 | 1.2 | Layer redundancy via CKA |
| HeadClusterer | 1 | 1.2 | Attention head functional clustering |
| JournalArchitect | 1 | 1.3 | Design hypothesis journal schema |
| JournalWriter | 1 | 1.3 | Integrate journal logging into loop |
| JournalReader | 1 | 1.3 | Query interface for journal entries |
| FailureExtractor | 2 | 2.1 | Parse rejected experiments |
| FailureClusterer | 2 | 2.1 | Cluster failure patterns |
| ConstraintGenerator | 2 | 2.1 | Convert clusters to negative constraints |
| ConstraintValidator | 2 | 2.1 | Back-test constraint quality |
| ScaleConfigDeriver | 2 | 2.2 | Generate multi-scale configs |
| ScaleRunner | 2 | 2.2 | Multi-scale training harness |
| ScalingCurveFitter | 2 | 2.2 | Power-law extrapolation |
| ScaleGateIntegrator | 2 | 2.3 | Wire scale gate into acceptance |
| ScaleGateLogger | 2 | 2.3 | Log scale-gate decisions |
| DiffParser | 3 | 3.1 | AST-level modification decomposition |
| ComponentIsolator | 3 | 3.1 | Generate isolated component patches |
| AblationOrchestrator | 3 | 3.2 | Leave-one-out ablation runner |
| AblationBudgeter | 3 | 3.2 | Compute budget management |
| ComponentStripper | 3 | 3.3 | Remove neutral/negative components |
| AblationJournalWriter | 3 | 3.3 | Extended journal with ablation data |
| CriticPromptDesigner | 4 | 4.1 | Design critic agent prompts |
| MetricImplementer | 4 | 4.1 | Code-gen for proposed metrics |
| MetricRegistry | 4 | 4.1 | Metric catalog management |
| MetricCorrelator | 4 | 4.2 | Compute metric predictive power |
| MetricPromoter | 4 | 4.2 | Promote/retire metrics |
| ContextBudgetManager | 4 | 4.2 | Manage research agent context |
| LoopIntegrator | 4 | 4.3 | Wire all Phase 4 into main loop |
| SafetyGuard | 4 | 4.3 | Guardrails against metric proliferation |
| PipelineMonitor | 4 | 4.3 | Monitoring dashboard |
| EndToEndValidator | Integration | — | 20-iteration full pipeline test |
| RegressionTester | Integration | — | 50-iteration A/B comparison |
| DocumentationAgent | Integration | — | Operator documentation |
