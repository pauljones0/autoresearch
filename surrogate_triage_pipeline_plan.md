# Literature-Informed Surrogate Triage Pipeline — Multi-Phase Implementation Plan

**Source:** Synergy B from `ideas_v2.md` (Risk 3/7, Reward 6/7)
**Status:** NOT IMPLEMENTED
**Depends on:** Model Scientist Pipeline (Synergy C) — **IMPLEMENTED**
**Goal:** Build a funnel that ingests techniques from arXiv broadly, scores them cheaply with a surrogate model trained on Model Scientist data, and routes the best candidates through the existing diagnostic → scale-gate → ablation → journal pipeline — breaking the LLM knowledge ceiling while inheriting all of Synergy C's scientific rigor.

---

## Dependency Map: What Synergy C Provides

The Model Scientist Pipeline (`model_scientist/`) provides the following infrastructure that this plan builds on directly:

| Synergy C Component | Location | How Surrogate Triage Uses It |
|---|---|---|
| `ModelScientistPipeline` | `model_scientist/pipeline.py` | Entry point for evaluating paper-sourced candidates — they go through the same scale-gate → train → ablate → journal pipeline as internal modifications |
| `JournalReader` | `model_scientist/journal/reader.py` | Training data source for the surrogate — every historical experiment is a (diff, delta) pair |
| `FailureExtractor.extract_features_vector()` | `model_scientist/failure_mining/extractor.py` | 23-element feature vectors augment diff embeddings for richer surrogate input |
| `NegativeConstraint` list | `model_scientist/failure_mining/constraints.py` | Pre-filters paper techniques before they reach the surrogate — no point scoring a technique that matches a known failure pattern |
| `DiagnosticsReport` | `model_scientist/diagnostics/schemas.py` | Informs which paper techniques are relevant to the model's *current* bottleneck, not just generally promising |
| `ScaleGate` | `model_scientist/scaling/gate.py` | Paper-sourced candidates that pass surrogate scoring still go through scale-gate before full evaluation |
| `AblationOrchestrator` | `model_scientist/ablation/orchestrator.py` | Successful paper-sourced modifications are decomposed to isolate the novel contribution |
| `MetricRegistry` | `model_scientist/metrics/registry.py` | Evolved metrics provide additional features for surrogate training beyond raw diff embeddings |
| `SafetyGuard` | `model_scientist/integration/safety_guard.py` | Compute budget enforcement — paper evaluation shares the same budget pool |

---

## Phase 1: Paper Ingestion & Technique Extraction

**Gate:** Phase 1 is complete when the ingestion pipeline processes at least 50 arXiv papers, extracts structured technique descriptions from at least 30 of them, and a manual review of 10 extractions confirms >70% faithfulness to the source paper. Must run unattended for 7 consecutive days.

### Subphase 1.1: ArXiv Monitoring & Retrieval

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the monitor fetches new papers from cs.LG, cs.CL, and cs.AI daily, deduplicates against previously seen papers, and stores metadata in `papers_index.jsonl` for at least 3 consecutive days.

| Agent | Task |
|-------|------|
| **ArxivFetcher** | Implement a daily arXiv polling script using the arXiv API. Query cs.LG, cs.CL, cs.AI for new submissions. Store paper metadata (title, abstract, authors, venue, PDF URL, arXiv ID, categories) in `papers_index.jsonl`. Handle rate limits, pagination, and deduplication by arXiv ID. |
| **PaperFilter** | Build a relevance filter with two scoring layers. Layer 1 (keyword): score abstracts against terms related to training efficiency, architecture, optimization, regularization. Layer 2 (diagnostics-informed): read the latest `DiagnosticsReport` from `model_scientist/pipeline._last_diagnostics` and boost papers whose abstracts address the model's *current* diagnosed bottleneck (e.g., if attention entropy is collapsed, boost papers about attention diversity). Only papers above a combined relevance threshold proceed. Log filtered-out papers with reasons. |
| **PDFDownloader** | Download PDFs for relevant papers into `papers_cache/`. Implement retry logic, storage limits (configurable max cache size), and cleanup of papers older than N days. |

### Subphase 1.2: Technique Extraction

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the extractor produces valid `technique_descriptions.jsonl` entries for at least 20 papers, each containing: technique name, modification category (using the same categories as `FailureExtractor`: architecture, optimizer, hyperparameter, activation, initialization, regularization, scheduling, other), pseudo-code, reported improvement, and applicability conditions. Manual review of 5 extractions confirms accuracy.

| Agent | Task |
|-------|------|
| **PaperReader** | Build an LLM-based paper reading agent that extracts structured technique descriptions from PDFs. Output schema must align with Model Scientist categories — the `modification_category` field uses the same taxonomy as `FailureExtractor._classify_modification()` in `model_scientist/failure_mining/extractor.py` so that failure constraints can filter paper techniques directly. Extract: technique name, category, what it changes, pseudo-code, reported improvement (with baseline), applicability conditions (model scale, data type, architecture family). |
| **ExtractionValidator** | Validate extractions: check pseudo-code is parseable, reported improvements include baselines, applicability conditions are specific. Additionally, run each extraction against the active `NegativeConstraint` list from `model_scientist/failure_mining/constraints.py` — flag techniques that match a known failure pattern (e.g., "attention head pruning when gradient norms in layers 1-3 are below 0.01") so operators can see the conflict before GPU time is spent. |
| **TechniqueDeduplicator** | Detect when multiple papers describe the same technique. Cluster techniques by description similarity and merge duplicates. Also deduplicate against the existing `hypothesis_journal.jsonl` via `JournalReader.search()` — if the system has already tried a substantially similar modification internally, mark the technique as "already explored" with a link to the journal entry. |

### Subphase 1.3: Synthetic Diff Generation

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the diff generator produces at least 3 variant diffs per technique for 10 test techniques, at least 1 variant per technique applies cleanly to `train.py` and passes a 10-step smoke test, and the generated diffs are parseable by `DiffParser` in `model_scientist/ablation/diff_parser.py`.

| Agent | Task |
|-------|------|
| **DiffGenerator** | For each extracted technique, prompt an LLM to generate concrete code diffs against the current `train.py`. Generate 3-5 variants per technique. Critical requirement: diffs must be parseable by the existing `DiffParser` class in `model_scientist/ablation/diff_parser.py` so that successful paper-sourced modifications can be decomposed and ablated through the standard Phase 3 pipeline. Each variant must include a `modification_category` tag matching the extraction. |
| **DiffApplicabilityChecker** | Test each generated diff: apply to a copy of `train.py`, check for syntax errors, run a 10-step smoke test. Additionally, verify each diff is decomposable by calling `DiffParser.parse()` — diffs that produce 0 components or fail parsing are marked as "undecomposable" and deprioritized (they can still be evaluated but won't benefit from ablation). |
| **ConstraintPreFilter** | Before diffs proceed to surrogate scoring, run them against the active failure constraints from `ConstraintGenerator.format_for_prompt()`. For each diff, extract a synthetic `FailureFeatures` vector using `FailureExtractor.extract_features_vector()` (with the current diagnostics snapshot as context) and check if the vector falls within any known failure cluster. Diffs matching failure patterns are not discarded but receive a penalty score that the surrogate must overcome. Log all constraint matches for feedback analysis. |

---

## Phase 2: Surrogate Model Training on Model Scientist Data

**Gate:** Phase 2 is complete when the surrogate achieves >0.6 Spearman rank correlation between predicted and actual val_bpb deltas on a held-out test set of at least 20 modifications drawn from `hypothesis_journal.jsonl`. The surrogate must score a new diff in <1 second. The feature pipeline must incorporate at least 3 evolved metrics from `MetricRegistry`.

### Subphase 2.1: Feature Engineering from Model Scientist Data

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the feature pipeline produces a fixed-dimensional vector for any code diff that combines: (1) code embedding, (2) failure feature vector from `FailureExtractor`, and (3) active evolved metrics from `MetricRegistry`. Cosine similarity between semantically similar diffs is higher than dissimilar diffs on 5 manually curated test pairs.

| Agent | Task |
|-------|------|
| **EmbeddingModelSelector** | Evaluate candidate code embedding models (CodeBERT, StarEncoder, UniXcoder) on the task of embedding `train.py` diffs. Select based on embedding quality, inference speed, and memory footprint. Document selection rationale. |
| **DiffEmbedder** | Build the core embedding pipeline: take a unified diff, preprocess (normalize whitespace, strip comments), embed using the selected model, output a fixed-dimensional vector. Store embeddings in `diff_embeddings.npz`. |
| **FeatureEnricher** | Build the enrichment layer that combines three feature sources into the surrogate's input vector: (1) code embedding from `DiffEmbedder`, (2) 23-element failure feature vector from `FailureExtractor.extract_features_vector()` capturing the model's state at time of evaluation, (3) current values of all `active` metrics from `MetricRegistry` (via `MetricRegistry.get_active_metrics()`). The combined vector captures *what* the modification does (embedding), *what state the model is in* (failure features + evolved metrics), and enables the surrogate to learn context-dependent predictions (a technique that helps when attention is collapsed may hurt when attention is healthy). |
| **EmbeddingQualityMonitor** | Ongoing embedding quality checks after each batch. Verify semantic structure is maintained. Additionally, track whether the enrichment features (failure vectors, evolved metrics) add predictive signal by comparing surrogate accuracy with and without enrichment on a holdout set every retraining cycle. |

### Subphase 2.2: Surrogate Model Training

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the surrogate MLP trains on at least 50 (enriched_feature, val_bpb_delta) pairs sourced from `hypothesis_journal.jsonl` and achieves >0.5 rank correlation on a 20% held-out split. Training completes in <5 minutes on CPU.

| Agent | Task |
|-------|------|
| **JournalDataExtractor** | Build a data extraction pipeline from the implemented `JournalReader`. For each entry in `hypothesis_journal.jsonl`, extract: the `modification_diff` (for embedding), the `actual_delta` (label), the `diagnostics_summary` (for failure feature reconstruction), and the `tags` (for category metadata). Handle entries with missing fields gracefully (early journal entries may predate some enrichment features). Output: `surrogate_training_data.npz` with aligned feature vectors and labels. |
| **SurrogateTrainer** | Implement a lightweight MLP surrogate (2-3 hidden layers, <10K parameters). Input dimension = code_embedding_dim + 23 (failure features) + N_active_metrics. Train with MSE loss. Implement cross-validation, early stopping, and hyperparameter search. Save best model to `surrogate_model.pt`. Track which feature dimensions contribute most (feature importance via permutation) to inform `MetricCorrelator` about which evolved metrics are load-bearing for the surrogate. |
| **SurrogateEvaluator** | Evaluate surrogate accuracy using: rank correlation (Spearman's rho), calibration plots, worst-case analysis (how often does a bad modification rank in the top-5?). Additionally, compute separate accuracy for internally-generated vs. paper-sourced modifications once both are available — the surrogate may be less accurate on novel paper techniques, which informs cold-start thresholds. |

### Subphase 2.3: Cold-Start & Journal-Bootstrapped Warmup

**Status:** NOT IMPLEMENTED
**Gate:** Passes when the cold-start protocol handles three regimes: (1) journal has <50 entries — no surrogate filtering, all candidates evaluated, (2) journal has 50-200 entries — conservative threshold (top 50%), (3) journal has 200+ entries — full threshold (top 20%). Transitions are automatic and logged.

| Agent | Task |
|-------|------|
| **ColdStartManager** | Implement the cold-start protocol. Unlike the standalone plan, this version bootstraps from existing `hypothesis_journal.jsonl` — if the Model Scientist Pipeline has already accumulated 100+ experiments, the surrogate can skip cold-start entirely and begin scoring immediately. Query `JournalReader.count()` at startup to determine the regime. Log regime transitions. |
| **SurrogateRetrainer** | Periodic retraining after every M new datapoints (default 20). Rebuild `surrogate_training_data.npz` from the full journal (which continues growing from both internal and paper-sourced experiments). Track accuracy over retraining cycles. If a new retraining degrades accuracy (possible when paper-sourced data shifts the distribution), roll back to the previous model and flag for investigation. |
| **FeatureDriftDetector** | Monitor for feature drift: as `MetricRegistry` promotes/retires metrics (Phase 4 of Synergy C), the surrogate's input dimension changes. When a metric is retired, zero-fill that dimension and flag for retraining. When a metric is promoted, extend the input dimension and trigger retraining. Coordinate with `MetricPromoter` in `model_scientist/metrics/promoter.py` via an event hook. |

---

## Phase 3: Funnel Integration with Model Scientist Loop

**Gate:** Phase 3 is complete when paper-sourced candidates flow through the full funnel (arXiv → extract → embed → score → constraint-filter → scale-gate → train → ablate → journal) for at least 2 weekly cycles, the surrogate's top-5 selections include at least 1 modification that improves val_bpb, and all paper-sourced evaluations are recorded in `hypothesis_journal.jsonl` with a `source: "paper"` tag.

### Subphase 3.1: Scoring, Ranking & Constraint-Gated Queue

**Status:** NOT IMPLEMENTED
**Gate:** Passes when paper-sourced technique diffs are scored by the surrogate, ranked, filtered against failure constraints, and the top-N (configurable, default 5) are queued in `evaluation_queue.json`. The queue must handle priority ordering, deduplication (cosine similarity > 0.95), and constraint-match penalties.

| Agent | Task |
|-------|------|
| **SurrogateScoringPipeline** | Wire the full scoring flow: take applicable diffs from Phase 1.3, enrich features (Phase 2.1), score with surrogate (Phase 2.2), apply constraint penalties from `ConstraintPreFilter` (Phase 1.3), rank by adjusted score. Output a ranked queue in `evaluation_queue.json`. Each queue entry includes: diff, technique metadata, surrogate score, constraint matches (if any), and paper source info. |
| **QueueManager** | Manage the evaluation queue: insert new candidates, remove evaluated ones, handle priority (adjusted surrogate score), prevent duplicate evaluation (cosine similarity > 0.95 against both queue and `hypothesis_journal.jsonl` entries tagged `source: "paper"`), enforce max queue size. Expose queue state to `PipelineMonitor` in `model_scientist/integration/monitor.py` for dashboard display. |
| **EvaluationScheduler** | Schedule GPU evaluation of queued paper candidates alongside internally-generated modifications. Interface with `SafetyGuard` in `model_scientist/integration/safety_guard.py` for compute budget enforcement — paper evaluations draw from the same compute pool. Configurable split: default 30% of iterations for paper-sourced candidates, 70% for internal. When no paper candidates are queued, give 100% to internal. |

### Subphase 3.2: Pipeline Routing — Paper Candidates Through Model Scientist

**Status:** NOT IMPLEMENTED
**Gate:** Passes when a paper-sourced candidate is evaluated end-to-end through `ModelScientistPipeline.evaluate_modification()` — including scale-gate, training, ablation (if accepted), and journal logging — with the journal entry correctly tagged `source: "paper"` and `paper_id: <arXiv_id>`. At least 5 paper candidates must complete this route.

| Agent | Task |
|-------|------|
| **PaperCandidateRouter** | Route paper-sourced candidates into the existing `ModelScientistPipeline.evaluate_modification()` entry point. For each queued candidate, construct the call with: `modified_source` (diff applied to current train.py), `hypothesis` (from paper extraction: "Technique X from paper Y should improve Z because..."), `predicted_delta` (surrogate's prediction), `modification_diff` (the synthetic diff), `tags` including `["source:paper", "paper_id:<arxiv_id>", "surrogate_score:<score>"]`. The candidate then flows through the standard pipeline: scale-gate → train → accept/reject → ablate → journal. No special handling needed — the Model Scientist Pipeline treats it like any other modification. |
| **PaperJournalEnricher** | After evaluation, enrich the journal entry with paper-specific metadata: arXiv ID, paper title, authors, technique name, surrogate predicted delta vs. actual delta, extraction confidence score, and which diff variant was used (out of the 3-5 generated). This metadata lives in the journal entry's `tags` and a new `paper_metadata` field in the `diagnostics_summary`. Enables paper-source quality tracking in Phase 3.3. |
| **AblationInsightExtractor** | When a paper-sourced modification is accepted and ablated by `AblationOrchestrator`, extract insights specific to paper techniques: which component of the technique carried the value? Did the paper's reported improvement mechanism match the ablation-revealed mechanism? Log discrepancies — a technique that works for a different reason than the paper claims is still valuable but signals poor extraction. |

### Subphase 3.3: Three-Level Feedback System

**Status:** NOT IMPLEMENTED
**Gate:** Passes when evaluation results flow back to update all three levels: (1) surrogate retraining data (via `JournalDataExtractor`), (2) extraction quality scores per paper, and (3) paper source quality tracker with data on at least 5 author/venue/category sources.

| Agent | Task |
|-------|------|
| **SurrogateFeedbackLoop** | After each paper-sourced evaluation, the result automatically enters `hypothesis_journal.jsonl` (via the standard `JournalWriter`). The `SurrogateRetrainer` (Phase 2.3) picks it up on the next retraining cycle. Additionally, compute and log the surrogate's prediction error for this candidate — large errors (predicted success, actual failure or vice versa) indicate the surrogate is poorly calibrated for paper-sourced techniques and may need architecture changes or more paper-sourced training data. |
| **ExtractionQualityTracker** | For each paper, compare: (1) the surrogate's prediction (based on the extraction → synthetic diff → embedding), (2) the actual val_bpb delta from GPU evaluation. Large prediction errors may indicate poor extraction (the synthetic diff didn't faithfully capture the technique) rather than surrogate failure. Compute per-paper extraction quality scores. Feed high-error papers back to `PaperReader` with the actual result and the diff that was tested, so the extraction prompt can be refined. |
| **PaperSourceTracker** | Build `paper_source_quality.json`: track running success rates across dimensions — author, first author institution, venue/workshop, arXiv primary category, technique category (using the same taxonomy as `FailureExtractor`), and publication month. A "success" is a paper-sourced modification that was accepted by the Model Scientist Pipeline (survived scale-gate AND had positive ablation contribution). Update after every paper-sourced evaluation. |
| **FailureMiningBridge** | When paper-sourced candidates are rejected, feed them into the existing `FailureExtractor` and `FailureClusterer` in `model_scientist/failure_mining/`. Paper-sourced failures enrich the failure pattern database with a broader distribution of modification types than internally-generated failures alone — the whole point of the paper pipeline is to surface *novel* techniques, so their failure modes are likely novel too. Tag paper-sourced failure patterns distinctly so `ConstraintGenerator` can produce constraints like "Techniques from recent MoE papers have failed 6/7 times on models below 256 width." |

---

## Phase 4: Ingestion Intelligence & Cross-System Optimization

**Gate:** Phase 4 is complete when the ingestion filter demonstrably prioritizes papers addressing the model's current diagnosed bottleneck (via `DiagnosticsReport`), source quality bias shifts extraction rates toward historically productive sources, and the surrogate's accuracy on paper-sourced candidates is within 80% of its accuracy on internally-generated candidates. The `MetricRegistry` must contain at least 1 metric derived from surrogate triage data.

### Subphase 4.1: Diagnostics-Driven Ingestion

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when the paper relevance filter dynamically adjusts based on the latest `DiagnosticsReport`: papers addressing the current top bottleneck receive a measurably higher relevance score than papers addressing non-bottleneck topics, and this shift is visible in `papers_index.jsonl` scoring.

| Agent | Task |
|-------|------|
| **DiagnosticsIngestionLinker** | Connect `PaperFilter` (Phase 1.1) to the live diagnostics system. Before each daily ingestion cycle, read `model_scientist/pipeline._last_diagnostics` and extract the top-3 bottlenecks (e.g., "attention entropy collapsed in heads 4,7", "gradient vanishing in layers 1-3", "rare token loss 3x higher than top-1k"). Convert these into dynamic keyword boosts for the relevance filter. Papers whose abstracts address the current bottleneck get a relevance boost proportional to the bottleneck's severity (as measured by the diagnostic metric's deviation from healthy range). |
| **BottleneckMapper** | Build a mapping from diagnostic bottleneck types to paper search terms. Examples: `attention_entropy_collapse` → ["attention diversity", "head redundancy", "multi-head pruning", "attention routing"], `gradient_vanishing_early_layers` → ["residual connections", "gradient flow", "initialization scheme", "normalization placement"]. The mapping is static initially but can be extended by the `CriticAgent` in `model_scientist/metrics/critic.py` as new bottleneck types are discovered. |
| **RelevanceCalibrator** | Calibrate the dynamic relevance boost so it doesn't overwhelm the base keyword score. Run a retrospective analysis: for historical paper-sourced evaluations, would diagnostics-informed relevance have ranked the successful papers higher? If not, the boost is miscalibrated. Adjust coefficients and re-test. |

### Subphase 4.2: Source Quality Bias

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when papers from top-quartile sources (by historical success rate) are extracted at a higher rate than bottom-quartile sources, and this bias is reflected in `papers_index.jsonl` relevance scores. The bias must not hard-filter any source — even low-quality sources should occasionally produce candidates.

| Agent | Task |
|-------|------|
| **IngestionBiasAgent** | Use `paper_source_quality.json` (Phase 3.3) to bias the ingestion filter. Papers from historically productive sources get an additive relevance boost. The boost is proportional to the source's success rate but capped to prevent the filter from becoming a closed loop (always reading the same 5 authors). Minimum relevance for any paper from any source: 10% of max relevance score, ensuring exploration of new sources. |
| **SourceDiversityEnforcer** | Prevent source bias from collapsing diversity. Track the distribution of extracted papers across sources. If any single author/venue/category exceeds 30% of total extractions in a weekly cycle, throttle that source's boost. The system should learn *preferences*, not *monopolies*. Alert if source diversity drops below a configurable threshold. |
| **NewSourceScout** | Actively seek papers from sources not yet represented in `paper_source_quality.json`. Allocate a fixed percentage (default 15%) of extraction slots to papers from unseen authors/venues. This is the exploration arm of the ingestion bias — without it, the system would never discover new productive sources. |

### Subphase 4.3: Cross-System Metric & Feedback Evolution

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when at least 1 new metric derived from surrogate triage data has been proposed to `CriticAgent`, implemented via `MetricImplementer`, and entered the `MetricRegistry` evaluation cycle. The `MetaLearningMonitor` shows trends at all four learning levels (model, surrogate, ingestion, metrics).

| Agent | Task |
|-------|------|
| **SurrogateMetricProposer** | Propose new diagnostic metrics derived from surrogate triage patterns. Examples: "surrogate_prediction_confidence" (how confident the surrogate is about the current model state — low confidence = the model has entered an unexplored regime), "paper_technique_novelty_score" (embedding distance of the best paper-sourced diff from all historically attempted diffs — high novelty = the pipeline is surfacing genuinely new ideas). Submit proposals to `CriticAgent.propose_metrics()` in `model_scientist/metrics/critic.py` for evaluation through the standard metric evolution pipeline. |
| **MetaLearningMonitor** | Track the four-level learning system's health: (1) Is val_bpb improving? (via `JournalReader.success_rate()`), (2) Is the surrogate's accuracy improving? (via `SurrogateEvaluator` reports), (3) Is the ingestion filter's precision improving? (successful papers / total extracted), (4) Are evolved metrics improving the surrogate? (feature importance trends). Produce a weekly health report showing trends at all four levels. |
| **PipelineReporter** | Generate a human-readable weekly report consolidating data from both the Model Scientist Pipeline and the Surrogate Triage Pipeline: papers ingested, techniques extracted, constraint pre-filter rejections, surrogate scores, GPU evaluations run (paper vs. internal), modifications accepted, ablation insights from paper techniques, source quality updates, metric evolution events, and diagnostics-driven relevance shifts. Output as `weekly_report.md`. Also push key metrics to `PipelineMonitor` in `model_scientist/integration/monitor.py` for inclusion in the HTML dashboard. |

---

## Phase 5: Steady-State Operation & Self-Improvement

**Gate:** Phase 5 is complete when the full pipeline has operated autonomously for 8 consecutive weeks, the paper-sourced acceptance rate is within 50% of the internal acceptance rate (papers are harder to implement correctly, so a gap is expected), the surrogate achieves >0.6 rank correlation on paper-sourced candidates specifically, and the system has surfaced at least 3 techniques the LLM would not have proposed independently (verified by checking they don't appear in the LLM's training data distribution via journal analysis).

### Subphase 5.1: Surrogate Specialization

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when the surrogate maintains separate calibration for internal vs. paper-sourced candidates, and the threshold for paper candidates is independently tunable without affecting internal scoring.

| Agent | Task |
|-------|------|
| **DualCalibrationManager** | Implement separate calibration curves for internally-generated vs. paper-sourced candidates. Paper techniques have higher implementation variance (the synthetic diff may not faithfully capture the technique), so the surrogate's predictions are noisier. Maintain two threshold values: one for internal candidates (tighter, since the surrogate is well-calibrated on these) and one for paper candidates (looser, allowing more through to compensate for extraction noise). |
| **PaperSpecificRetraining** | Implement optional paper-specific surrogate fine-tuning: after accumulating 50+ paper-sourced evaluations, train a small adapter layer that adjusts the surrogate's predictions for paper-sourced diffs. This compensates for the systematic bias introduced by synthetic diff generation (LLM-generated diffs may have stylistic patterns that differ from real autoresearch modifications). |

### Subphase 5.2: Extraction Quality Feedback Loop

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when the extraction prompt has been refined at least 3 times based on evaluation feedback, and extraction quality (measured by surrogate prediction error on paper-sourced candidates) improves monotonically across refinements.

| Agent | Task |
|-------|------|
| **ExtractionPromptEvolver** | Analyze patterns in extraction failures: when paper-sourced candidates have high surrogate prediction error, examine the extraction → diff → evaluation chain to identify where information was lost. Common failure modes: (1) extraction missed a critical detail (e.g., "only works with pre-norm architecture"), (2) diff generator misinterpreted the pseudo-code, (3) technique is fundamentally incompatible with `train.py`'s architecture. Refine the `PaperReader` prompt to address the most common failure modes. Track extraction quality across prompt versions. |
| **DiffQualityAnalyzer** | Compare the N diff variants generated per technique. When one variant succeeds and others fail, analyze what the successful variant got right. Build a corpus of "good diff" patterns and "bad diff" patterns. Feed these patterns back to `DiffGenerator` to improve future diff generation. Cross-reference with `AblationInsightExtractor` (Phase 3.2) — when ablation reveals that only one component of a paper technique is valuable, document which part of the paper description mapped to the valuable component. |

### Subphase 5.3: Novel Technique Detection & Impact Tracking

**STATUS:** NOT IMPLEMENTED
**Gate:** Passes when the system can distinguish genuinely novel paper-sourced techniques (not in the LLM's natural proposal distribution) from techniques the LLM would have proposed anyway, and at least 2 accepted modifications are classified as "genuinely novel."

| Agent | Task |
|-------|------|
| **NoveltyClassifier** | For each accepted paper-sourced modification, estimate whether it was "genuinely novel" — would the internal research LLM have proposed it independently? Method: embed the paper technique's diff and compute its nearest-neighbor distance to all internally-generated diffs in `hypothesis_journal.jsonl`. Techniques with high minimum distance are classified as novel. Also query the research LLM: "Would you have proposed this modification without being told about this paper?" and compare with the embedding-based classification. |
| **ImpactTracker** | Track the long-term impact of paper-sourced modifications. A paper technique accepted at iteration N may enable future improvements at iterations N+10, N+20 that wouldn't have been possible otherwise (e.g., an architectural change that opens new optimization avenues). Measure: val_bpb improvement rate in the 20 iterations following a paper-sourced acceptance vs. the baseline rate. This captures the "stepping stone" value of novel techniques beyond their immediate val_bpb delta. |
| **KnowledgeCeilingMonitor** | Track whether the paper pipeline is actually breaking the LLM knowledge ceiling. Metric: over rolling 50-iteration windows, what fraction of accepted modifications are paper-sourced? If this fraction trends upward over time, the LLM's internal knowledge is being exhausted and the paper pipeline is becoming increasingly critical. If it trends toward zero, the LLM is sufficient and the paper pipeline's value is marginal. Report this trend to operators as a key strategic indicator. |

---

## Full Pipeline Integration

**Gate:** The Literature-Informed Surrogate Triage Pipeline is **IMPLEMENTED** when all five phases pass their gates AND an end-to-end demonstration shows the following sequence completing for 4 consecutive weeks without human intervention:

1. Daily: ArXiv papers fetched, filtered by keyword + diagnostics-informed relevance + source quality bias (Phase 1 + 4)
2. Daily: Techniques extracted, deduplicated against journal, constraint-pre-filtered (Phase 1)
3. Daily: Synthetic diffs generated, applicability-checked, `DiffParser`-validated (Phase 1)
4. Daily: Diffs enriched (embedding + failure features + evolved metrics), scored by surrogate, queued (Phase 2 + 3)
5. Per-iteration: Top paper candidates routed through `ModelScientistPipeline.evaluate_modification()` — scale-gate → train → accept/reject → ablate → journal (Phase 3)
6. Per-evaluation: Results feed back to surrogate training data, extraction quality tracker, paper source quality tracker, and failure mining system (Phase 3)
7. Weekly: Ingestion bias updated, surrogate retrained, metrics evolved, weekly report generated (Phase 4)
8. Ongoing: Surrogate calibration maintained, extraction prompts refined, novelty and impact tracked (Phase 5)

| Agent | Task |
|-------|------|
| **EndToEndValidator** | Run the full pipeline for 4 weekly cycles. Verify all stages execute, data flows correctly between surrogate triage and Model Scientist systems, no stage silently drops candidates, and journal entries are correctly tagged with paper metadata. |
| **BaselineComparator** | Compare val_bpb improvement rate with vs. without the paper pipeline over a 100-iteration window using the same `ModelScientistPipeline` configuration. The pipeline should surface at least 1 technique the LLM would not have proposed independently (verified by `NoveltyClassifier`). Also compare: surrogate accuracy, failure pattern diversity, and metric evolution rate between the two conditions. |
| **IntegrationDocumentationAgent** | Write operator documentation covering: how the surrogate triage pipeline interfaces with each Model Scientist component, configuration guide (relevance thresholds, surrogate retrain frequency, compute split, constraint penalty weights), monitoring procedures via the existing `PipelineMonitor` dashboard, how to manually add papers or techniques, and troubleshooting (API rate limits, extraction quality drops, surrogate drift, feature dimension changes from metric evolution). |

---

## Agent Summary

| Agent Name | Phase | Subphase | Role |
|---|---|---|---|
| ArxivFetcher | 1 | 1.1 | Daily arXiv polling and metadata storage |
| PaperFilter | 1 | 1.1 | Keyword + diagnostics-informed relevance scoring |
| PDFDownloader | 1 | 1.1 | PDF retrieval and cache management |
| PaperReader | 1 | 1.2 | LLM-based extraction aligned with FailureExtractor categories |
| ExtractionValidator | 1 | 1.2 | Validate extractions + check against failure constraints |
| TechniqueDeduplicator | 1 | 1.2 | Dedup against other papers AND hypothesis journal |
| DiffGenerator | 1 | 1.3 | Generate DiffParser-compatible synthetic diffs |
| DiffApplicabilityChecker | 1 | 1.3 | Smoke-test + DiffParser decomposability check |
| ConstraintPreFilter | 1 | 1.3 | Failure constraint matching via FailureExtractor vectors |
| EmbeddingModelSelector | 2 | 2.1 | Evaluate and select code embedding model |
| DiffEmbedder | 2 | 2.1 | Core diff → vector embedding |
| FeatureEnricher | 2 | 2.1 | Combine embedding + failure features + evolved metrics |
| EmbeddingQualityMonitor | 2 | 2.1 | Ongoing embedding + enrichment quality checks |
| JournalDataExtractor | 2 | 2.2 | Build surrogate training data from hypothesis journal |
| SurrogateTrainer | 2 | 2.2 | Train MLP surrogate on enriched features |
| SurrogateEvaluator | 2 | 2.2 | Evaluate surrogate accuracy (internal vs. paper split) |
| ColdStartManager | 2 | 2.3 | Journal-bootstrapped warmup regime |
| SurrogateRetrainer | 2 | 2.3 | Periodic retraining with rollback |
| FeatureDriftDetector | 2 | 2.3 | Handle MetricRegistry promotions/retirements |
| SurrogateScoringPipeline | 3 | 3.1 | Full scoring flow with constraint penalties |
| QueueManager | 3 | 3.1 | Priority queue with journal-aware dedup |
| EvaluationScheduler | 3 | 3.1 | GPU allocation via SafetyGuard budget |
| PaperCandidateRouter | 3 | 3.2 | Route candidates into ModelScientistPipeline |
| PaperJournalEnricher | 3 | 3.2 | Add paper metadata to journal entries |
| AblationInsightExtractor | 3 | 3.2 | Analyze paper technique ablation results |
| SurrogateFeedbackLoop | 3 | 3.3 | Prediction error tracking for paper candidates |
| ExtractionQualityTracker | 3 | 3.3 | Per-paper extraction quality scores |
| PaperSourceTracker | 3 | 3.3 | Author/venue/category success rates |
| FailureMiningBridge | 3 | 3.3 | Feed paper failures into FailureClusterer |
| DiagnosticsIngestionLinker | 4 | 4.1 | Connect PaperFilter to live DiagnosticsReport |
| BottleneckMapper | 4 | 4.1 | Map diagnostic bottlenecks to search terms |
| RelevanceCalibrator | 4 | 4.1 | Calibrate diagnostics boost coefficients |
| IngestionBiasAgent | 4 | 4.2 | Source quality relevance boost |
| SourceDiversityEnforcer | 4 | 4.2 | Prevent source bias collapse |
| NewSourceScout | 4 | 4.2 | Allocate slots for unseen sources |
| SurrogateMetricProposer | 4 | 4.3 | Propose triage-derived metrics to CriticAgent |
| MetaLearningMonitor | 4 | 4.3 | Four-level learning health tracking |
| PipelineReporter | 4 | 4.3 | Unified weekly report + dashboard integration |
| DualCalibrationManager | 5 | 5.1 | Separate thresholds for internal vs. paper candidates |
| PaperSpecificRetraining | 5 | 5.1 | Adapter layer for paper-sourced diff bias |
| ExtractionPromptEvolver | 5 | 5.2 | Refine PaperReader prompts from evaluation feedback |
| DiffQualityAnalyzer | 5 | 5.2 | Analyze good/bad diff variant patterns |
| NoveltyClassifier | 5 | 5.3 | Distinguish genuinely novel paper techniques |
| ImpactTracker | 5 | 5.3 | Long-term stepping-stone value measurement |
| KnowledgeCeilingMonitor | 5 | 5.3 | Track LLM knowledge exhaustion trend |
| EndToEndValidator | Integration | — | 4-week full pipeline test |
| BaselineComparator | Integration | — | 100-iteration A/B with novelty verification |
| IntegrationDocumentationAgent | Integration | — | Operator docs for cross-system operation |

---

## Cross-Reference: Data Flow Between Systems

```
                        ┌─────────────────────────────────────────────────┐
                        │          MODEL SCIENTIST PIPELINE               │
                        │              (IMPLEMENTED)                      │
                        │                                                 │
  ArXiv ──→ PaperFilter ──────── reads ──────→ DiagnosticsReport         │
              │                                     │                     │
              ▼                                     ▼                     │
         PaperReader ─── categories from ──→ FailureExtractor            │
              │                                     │                     │
              ▼                                     ▼                     │
       ConstraintPreFilter ◄── constraints ── ConstraintGenerator        │
              │                                                           │
              ▼                                                           │
        DiffGenerator ─── validated by ──→ DiffParser (ablation)         │
              │                                                           │
              ▼                                                           │
        FeatureEnricher ◄── features from ── FailureExtractor            │
              │          ◄── metrics from ── MetricRegistry              │
              ▼                                                           │
        SurrogateScorer                                                   │
              │                                                           │
              ▼                                                           │
    PaperCandidateRouter ──→ ModelScientistPipeline.evaluate_modification│
                                    │                                     │
                              ┌─────┴──────┐                             │
                              ▼            ▼                             │
                          ScaleGate    AblationOrchestrator              │
                              │            │                             │
                              ▼            ▼                             │
                          JournalWriter ◄──┘                             │
                              │                                          │
                              ▼                                          │
                    hypothesis_journal.jsonl ──→ SurrogateRetrainer      │
                              │                                          │
                              ▼                                          │
                    FailureMiningBridge ──→ FailureClusterer             │
                                                                         │
                        └─────────────────────────────────────────────────┘
```
