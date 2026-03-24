# Autoresearch Ideas v2: Synergy Explorations

This document explores five synergy clusters identified across the original 23 ideas in `ideas.md`. Rather than re-listing every idea, it focuses on how specific ideas combine into systems greater than the sum of their parts — and where external research validates or challenges feasibility.

---

## Synergy A: Adaptive Bandit with Simulated Annealing (Ideas 3 + 16)

> **Risk: 2/7** — Minimal infrastructure, degrades gracefully to greedy search, one JSON file of state.
> **Reward: 5/7** — Principled exploration-exploitation with per-category risk tolerance. Moderate but reliable gains.

**Core Insight:** Thompson Sampling (Idea 3) decides *what type* of modification to try. Simulated annealing (Idea 16) decides *whether to accept a risky result*. Alone, each solves half the exploration-exploitation problem. Together, they create a fully adaptive search strategy that controls both proposal generation and acceptance criteria.

### Why the Pairing Works

Thompson Sampling maintains posterior distributions over modification categories (architecture changes, optimizer swaps, hyperparameter tweaks, etc.) and samples from them to decide what to try next. But it says nothing about how to handle the result — it uses the same binary accept/reject criterion as baseline autoresearch. This means a promising but immature category (say, "attention mechanism modifications" with a wide posterior) gets explored more often, but individual experiments within that category are still evaluated greedily. A modification that temporarily worsens val_bpb to escape a local optimum is still rejected, even if the bandit correctly identified attention changes as high-potential.

Simulated annealing fixes this by making acceptance probabilistic. But vanilla annealing applies a single global temperature to all modifications regardless of type. This is suboptimal: a well-understood category (narrow posterior, many data points) should use a low temperature and only accept strict improvements, while an uncertain category (wide posterior, few data points) should tolerate more risk. The information is right there in the bandit's posteriors — it just needs to flow into the acceptance criterion.

### Combined Architecture: Per-Arm Annealing

The combined system maintains, for each modification category arm:
- A Beta(α, β) posterior tracking success rate (from Thompson Sampling)
- A per-arm temperature T_arm derived from the posterior's variance: `T_arm = T_base × σ(posterior)`, where σ is the standard deviation of the Beta distribution

High-variance arms (uncertain categories) get higher temperatures — the system takes more risks on modification types it hasn't explored enough. Low-variance arms (well-characterized categories) get lower temperatures — it only accepts proven improvements. As the posterior narrows through accumulated evidence, the temperature naturally cools for that arm. This is annealing, but driven by evidence rather than an arbitrary schedule.

Adaptive reheating also becomes per-arm: if a specific category hasn't produced an accepted modification in K consecutive attempts *despite* the bandit continuing to select it (high posterior mean but recent failures), the arm's temperature spikes. This signals that the category may have entered a new regime where old rules don't apply — perhaps an architectural change shifted the optimization landscape for hyperparameter tuning.

### Implementation Sketch

A single JSON state file (`strategy_state.json`) tracks everything:

```json
{
  "arms": {
    "architecture": {"alpha": 12, "beta": 8, "temperature": 0.015, "consecutive_failures": 0},
    "optimizer": {"alpha": 5, "beta": 15, "temperature": 0.008, "consecutive_failures": 3},
    "hyperparameter": {"alpha": 20, "beta": 30, "temperature": 0.004, "consecutive_failures": 0},
    ...
  },
  "global_iteration": 147,
  "T_base": 0.02
}
```

Each iteration:
1. Sample from each arm's Beta posterior → select the arm with the highest sample
2. Prompt the LLM to propose a modification of that type
3. Train and evaluate → compute val_bpb delta
4. Accept if `delta < 0` (improvement) OR `random() < exp(-delta / T_arm)` (annealing)
5. Update the selected arm: α += 1 on success, β += 1 on failure; recompute T_arm from posterior variance
6. If consecutive_failures > K, multiply T_arm by reheat_factor

The overhead is negligible — a few arithmetic operations per iteration. The system degrades gracefully: if all posteriors collapse, it behaves like greedy search with category preferences; if temperatures are all zero, it ignores the annealing. The combined system strictly dominates either component alone because it uses more information.

### Expected Outcome

The bandit prevents wasting iterations on exhausted categories. The annealing prevents getting stuck within a category. Together, the system should plateau later and reach lower val_bpb than either approach alone, with the per-arm temperature providing a principled mechanism for category-specific risk tolerance rather than relying on a human-tuned global schedule.

---

## Synergy B: Literature-Informed Surrogate Triage (Ideas 6 + 22)

> **Risk: 3/7** — Surrogate cold-start period, noisy paper extraction, hallucinated implementation details.
> **Reward: 6/7** — Breaks the LLM knowledge ceiling; introduces genuinely novel techniques the system would never generate alone.

**Core Insight:** The paper reading pipeline (Idea 6) generates a firehose of candidate techniques from arXiv. The surrogate model (Idea 22) provides a cheap scoring function. Together, they create an efficient funnel: ingest broadly, filter cheaply, evaluate expensively only the best candidates.

### The Funnel Architecture

**Stage 1 — Ingestion (Idea 6):** An LLM monitors arXiv daily, reads papers from cs.LG/cs.CL/cs.AI, and extracts structured technique descriptions: what the technique changes, pseudo-code, reported improvement, and applicability conditions. This produces perhaps 20-50 candidate techniques per week — far more than the training loop can evaluate at 12 experiments per hour.

**Stage 2 — Embedding:** Each extracted technique is converted into a synthetic code diff — the LLM generates what the modification to train.py would look like if the technique were applied. This diff is embedded using a code embedding model (CodeBERT, StarEncoder) into the same vector space the surrogate was trained on.

**Stage 3 — Surrogate Scoring (Idea 22):** The surrogate MLP predicts the expected val_bpb delta for each embedded diff. Techniques are ranked by predicted improvement. Only the top-N (say, top 3-5 per week) enter the actual training evaluation queue.

**Stage 4 — GPU Evaluation:** The selected techniques are tested through the standard autoresearch loop. Results feed back into both the surrogate's training set and the paper quality model.

### Why the Combination Multiplies Value

Without the surrogate, the paper pipeline has a scaling problem: it generates more candidates than the training loop can process, forcing random or heuristic selection. Without the paper pipeline, the surrogate is limited to scoring variations of what the LLM already knows — it can rank 10 LLM-generated proposals but can't introduce genuinely novel ideas from recent research.

Together, the paper pipeline provides breadth (novel ideas the LLM's training data doesn't contain) and the surrogate provides depth (cheap evaluation to focus expensive GPU time on the most promising novel ideas). The surrogate also improves faster in this configuration because paper-sourced techniques are more diverse than LLM-generated proposals, providing better coverage of the embedding space for the surrogate's training data.

### The Feedback Loop

A second-order benefit emerges: the system tracks which papers, authors, venues, and technique categories produce high-surrogate-scoring and (more importantly) actually-successful modifications. Over time, this builds a meta-model of research quality — a data-driven answer to "which arXiv papers are worth reading for practical training improvements?" This meta-model can then bias the ingestion stage, spending more extraction effort on papers from historically productive sources.

This creates a three-level learning system:
1. The training loop learns to train better models
2. The surrogate learns to predict which modifications will help
3. The ingestion filter learns which papers produce useful modifications

### Implementation Notes

The paper extraction LLM can be the same model as the research agent, running during GPU-bound training downtime. The surrogate needs ~50-100 datapoints before it's useful, so the first few weeks run without filtering (pure Idea 6). The synthetic diff generation is the trickiest part — the LLM must translate abstract technique descriptions into concrete train.py modifications, which is lossy and may hallucinate implementation details. Mitigation: generate multiple diff variants per technique and score all of them, increasing the chance that at least one captures the technique correctly.

Cold-start is handled naturally: the surrogate trains on internally-generated modifications first, then paper-sourced ones start flowing in as additional training data. The system never depends on the surrogate being accurate from day one.

---

## Synergy C: The Model Scientist Pipeline (Ideas 7, 10, 12 + variants of 1, 2, 4, 17, 21)

> **Risk: 5/7** — Highest complexity; 7 interacting stages, significant compute overhead from ablations and scaling tests, metric evolution could introduce instability.
> **Reward: 7/7** — Full scientific method loop with compounding returns. Each iteration makes the system a better scientist, not just a better optimizer.

**Core Insight:** This is the largest synergy cluster, combining eight ideas into a full scientific method loop for automated ML research. The model doesn't just optimize — it diagnoses, hypothesizes, experiments, ablates, documents, and learns from failure, with its measurement tools evolving alongside its understanding.

### The Seven-Stage Loop

**Stage 1 — Diagnose (Ideas 10 + 12: Gradient Diagnostics + Interpretability)**

After each training run, the system produces a rich diagnostic report. From Idea 10: per-layer gradient norms, activation statistics, loss decomposition by token type, attention entropy per head, dead neuron counts. From Idea 12: probing classifiers reveal what information is encoded where, attention pattern clustering identifies functional head types, CKA similarity across layers flags redundant computation.

These aren't just numbers — they're a medical chart for the model. "Gradients vanishing in layers 1-3, attention heads 4 and 7 have collapsed to near-uniform distributions, the model spends 60% of capacity on the top-5000 tokens and starves on rare vocabulary" is a diagnosis that points directly at interventions.

**Stage 2 — Dynamic Metric Generation (variant of Idea 1: Critic as Metric Designer)**

Here the critic agent takes a different role than in Idea 1's adversarial setup. Instead of trying to break modifications, the critic examines the diagnostic report and proposes *new diagnostic metrics* that would better capture the model's current bottleneck. Early in training, simple metrics (loss, gradient norm) suffice. As the model improves and obvious problems are fixed, the critic proposes increasingly subtle metrics: per-frequency-bucket calibration, attention pattern diversity indices, representation rank per layer, information-theoretic measures of layer utilization.

This is where the "variant" matters — the critic isn't adversarial, it's analytical. It asks: "Given what the diagnostics show, what *should* we be measuring that we aren't?" This continuously sharpens the system's ability to see what's wrong.

**Stage 3 — Metric Evolution (variant of Idea 2: Evolving the Evaluation)**

Not all critic-proposed metrics are useful. Stage 3 runs a lightweight meta-evaluation: which of the accumulated diagnostic metrics best predict whether a modification will succeed? Metrics that correlate strongly with modification success get promoted (higher weight in the diagnostic summary shown to the research agent). Metrics that don't predict anything get pruned. This is Idea 2's core concept — evolving the measurement apparatus — but applied to diagnostics rather than the primary objective function. Val_bpb remains the fixed ground truth; what evolves is the system's intermediate measurement toolkit.

**Stage 4 — Scale-Aware Intervention (Idea 7: Scaling Law Transfer)**

The research agent receives the evolved diagnostic summary and proposes a targeted intervention. Before committing to it, the system runs the modification at 2-3 additional small scales and fits a scaling curve. If the improvement extrapolates as durable at larger scales, proceed. If it shrinks toward zero, discard it as a small-scale artifact. This prevents wasting ablation compute (Stage 5) on modifications that won't survive scale-up.

**Stage 5 — Ablate (Idea 4: Causal Intervention Analysis)**

Modifications that survive scaling analysis are decomposed into independent components. Each component is ablated (removed) and the remaining modification is re-evaluated. This produces marginal contributions: "The activation function swap contributed +0.02 bpb, the initialization change was neutral, the width adjustment was -0.005 bpb." The neutral and negative components are stripped; only the causally validated components are kept.

**Stage 6 — Document (Idea 17: Hypothesis Journal)**

The full pipeline is recorded: initial diagnostics → critic-proposed metrics → the intervention hypothesis (with quantitative prediction) → scaling analysis results → ablation results → final verdict. This creates a structured scientific record far richer than "changed X, val_bpb went down by Y." Over hundreds of iterations, the journal becomes a dataset of ML experiments with diagnoses, hypotheses, predictions, interventions, and causal analyses — potentially publishable research.

**Stage 7 — Mine Failures (Idea 21: Failure Pattern Mining)**

Rejected modifications from Stages 4 and 5 (failed scaling, negative ablation components) are clustered into failure patterns. These become negative constraints: "Modifications targeting attention entropy when model width < 256 have failed 8 out of 9 times — avoid." The constraints feed back into Stage 1, focusing future diagnostics away from well-trodden dead ends.

### Compounding Returns

The power of this pipeline is that each stage improves the others over time:
- Better diagnostics (Stage 1) → more targeted interventions → higher acceptance rates
- Dynamic metrics (Stage 2) → the diagnostic report becomes more informative with each iteration
- Metric evolution (Stage 3) → noise is pruned, signal is amplified
- Scaling analysis (Stage 4) → fewer wasted ablation runs
- Ablation (Stage 5) → cleaner modifications with no dead weight
- Documentation (Stage 6) → the agent's context includes increasingly rich scientific history
- Failure mining (Stage 7) → the space of plausible modifications narrows toward productive regions

After 100 iterations, the system isn't just a better optimizer — it's a better *scientist*. Its measurement tools have evolved, its understanding of failure modes has deepened, and its documentation provides a growing body of evidence for future reasoning.

### Phased Rollout

This doesn't need to ship as one monolithic system. A practical rollout:
- **Phase 1:** Implement diagnostics (Stage 1) and documentation (Stage 6). Immediate value with minimal complexity.
- **Phase 2:** Add failure mining (Stage 7) and scaling analysis (Stage 4). These have the highest value-to-effort ratio.
- **Phase 3:** Add ablation (Stage 5). Requires more compute but provides causal understanding.
- **Phase 4:** Add the critic (Stage 2) and metric evolution (Stage 3). These are the most experimental components and benefit from having Phases 1-3 producing data first.

---

## Synergy D: GPU Kernel Creation — Optimizing Below the Abstraction Layer (Idea 18, expanded)

> **Risk: 6/7** — Silent numerical errors in generated kernels, hardware-specific brittleness, correctness verification is hard to get right.
> **Reward: 6/7** — Orthogonal to algorithmic gains (multiplicative stacking). Kernel-level wins like FlashAttention are among the largest practical speedups in modern ML.

**Core Insight:** Autoresearch operates at the Python/PyTorch level, but a large fraction of real-world training speedups come from kernel-level optimization — memory access patterns, operation fusion, hardware-specific implementations. Extending autoresearch to generate custom Triton kernels opens a fundamentally new optimization surface.

### What's Feasible Now

Triton (OpenAI's GPU programming language) has dramatically lowered the barrier to kernel writing. Unlike raw CUDA, Triton handles block-level parallelism and memory management semi-automatically, allowing the programmer to focus on the algorithm. Current LLMs can generate correct Triton kernels for:

- **Simple operation fusion:** Combining elementwise operations (activation + bias + dropout) into a single kernel, eliminating intermediate memory reads/writes
- **Custom attention variants:** FlashAttention-style tiled attention with modified scoring functions (e.g., ALiBi, RoPE variants)
- **Fused optimizer steps:** Combining gradient computation with weight updates to halve memory traffic
- **Quantized operations:** Mixed-precision matmuls with custom accumulation strategies

These are the "low-hanging fruit" — operations where the mathematical specification is clear and correctness can be verified numerically against the PyTorch reference implementation.

### What's Hard

- **Complex memory management:** Multi-stage pipelines where data must flow between shared memory, registers, and global memory in specific patterns. Subtle bugs cause silent numerical errors rather than crashes.
- **Multi-kernel coordination:** Operations spanning multiple kernel launches where synchronization and data layout must be co-optimized.
- **Hardware-specific tuning:** Optimal block sizes, warp configurations, and memory access patterns differ between GPU architectures (A100 vs H100 vs MI300X). A kernel that's 2x faster on A100 might be slower on H100.
- **Correctness verification at scale:** A kernel that's numerically equivalent on small inputs might diverge on large inputs due to floating-point ordering differences. Extensive testing across input shapes and dtypes is essential.

### External Validation: AlphaEvolve

Google DeepMind's AlphaEvolve (2025) provides strong evidence that AI-generated kernel optimization works at production scale. Using an evolutionary approach powered by Gemini, AlphaEvolve discovered an optimized kernel for a critical operation in Gemini's architecture that achieved a **23% speedup**, contributing to a 1% reduction in Gemini's overall training compute. AlphaEvolve also discovered an improved algorithm for 4x4 complex-valued matrix multiplication using 48 scalar multiplications — improving on Strassen's 1969 result.

The key design pattern from AlphaEvolve: pair LLM-based creative proposal with automated correctness verification and performance benchmarking. The LLM proposes kernel variants; an automated harness verifies numerical correctness (bitwise comparison to reference) and measures wall-clock performance. Only correct, faster kernels are accepted. This is directly applicable to autoresearch.

### Integration with the Broader System

Kernel optimization slots naturally into the Thompson Sampling bandit (Synergy A) as a new arm category. The bandit tracks kernel proposals alongside Python-level changes, learning when kernel optimization yields better returns than algorithmic changes. Early in autoresearch (when Python-level gains are abundant), the kernel arm's posterior will be wide and it'll be explored occasionally. As Python-level gains plateau, the bandit naturally shifts allocation toward kernel optimization — exactly when it becomes most valuable.

The diagnostic pipeline (Synergy C, Stage 1) also informs kernel optimization. GPU profiling data (kernel execution times, memory bandwidth utilization, occupancy) is already part of the diagnostic report. When diagnostics reveal that a specific operation is memory-bandwidth-bound with low occupancy, the system can propose a fused kernel targeting that exact bottleneck.

### Risk/Reward Assessment

**Risk:** Incorrect kernels can produce silent numerical errors that compound over training, leading to subtly degraded models without obvious failure signals. The correctness verification harness must be extremely robust — testing across multiple input shapes, dtypes, and edge cases.

**Reward:** Kernel-level optimization is where some of the largest practical speedups in modern ML have come from (FlashAttention: 2-4x, PagedAttention: 4-24x memory efficiency). These gains are *orthogonal* to algorithmic improvements — a better kernel and a better algorithm stack multiplicatively. If autoresearch can reliably generate even simple fused kernels, the wall-clock training efficiency gains would be substantial.

**Recommendation:** Start with the low-hanging fruit (elementwise fusion, fused optimizer steps) where correctness verification is straightforward. Build the correctness harness first, then expand scope gradually. Treat kernel generation as a specialized capability that activates when diagnostics identify a bottleneck, not as a default exploration strategy.

---

## Synergy E: Meta-Autoresearch — Can the System Improve Its Own Research Process? (Idea 20, deep feasibility analysis)

> **Risk: 4/7** — Bounded version is manageable; unbounded version is mathematically proven to degrade. Main risk is accidentally crossing from bounded into unbounded territory.
> **Reward: 5/7** — Human-designed harness defaults are almost certainly suboptimal. Even crude meta-optimization of prompts and acceptance criteria should yield real gains.

**Core Insight:** Meta-autoresearch — optimizing the optimization loop itself — is the logical endgame of autoresearch. The user's assessment that this is "ideal but near impossible" is partially correct: full recursive self-improvement faces fundamental mathematical limits, but *bounded* meta-optimization of specific harness components is feasible and already deployed in production systems.

### What External Research Shows

#### Systems That Work: Bounded Recursive Optimization

**Self-Taught Optimizer (STOP)** (Zelikman et al., Microsoft, COLM 2024, arXiv:2310.02304): A scaffolding program recursively improves itself using a fixed LLM (GPT-4). The framework generates multiple self-improvement strategies including beam search, genetic algorithms, and simulated annealing. Critical design choice: the LLM itself is fixed — only the generated code improves. This bounds the recursion and prevents entropy collapse.

**AlphaEvolve** (Google DeepMind, 2025, arXiv:2506.13131): An evolutionary coding agent powered by Gemini that discovers and optimizes algorithms. Pairs LLM creative generation with automated evaluation. Rediscovered state-of-the-art solutions 75% of the time across 50 open math problems and *improved* on the state of the art 20% of the time. Crucially, evaluation is always external and fixed — the system never modifies its own fitness function.

**RISE — Recursive Introspection** (NeurIPS 2024, arXiv:2407.18219): Frames prompt improvement as a multi-turn MDP. Shows that LLMs (Llama, Mistral) can improve their own outputs through multiple refinement turns without disrupting base capabilities. Evidence that shallow recursion (2-5 turns) is reliable; deeper recursion shows diminishing returns.

**Automatic Prompt Optimization** (multiple systems, 2024-2025): Google's Vertex AI Prompt Optimizer (NeurIPS 2024), ETGPO, GAAPO, and others demonstrate that LLMs can effectively optimize their own prompts through evolutionary and Bayesian approaches. This is directly relevant: autoresearch's system prompt is a meta-parameter that could be self-optimized.

#### Mathematical Limits: Why Unbounded Recursion Fails

**Entropy Decay in Self-Referential Loops** (arXiv:2601.05280, January 2026): This paper provides a mathematical proof that self-referential training loops exhibit entropy decay and variance amplification under finite sampling. The mutual information between model state and the true data distribution can only decrease or stay the same with each self-training iteration. Without external grounding (fresh authentic data), systems converge to "distorted and impoverished" distributions. The paper models this as a dynamical system on probability distributions and proves convergence to fixed points that are provably worse than the initial state.

**The Self-Reference Paradox** (Kumar, Future of Life Institute): An agent using its own reasoning to verify that reasoning system is fundamentally flawed — a self-referential paradox. An AI system cannot use its own reasoning to determine whether its reasoning is good. This is analogous to Godel's incompleteness theorems: a sufficiently powerful system cannot prove its own consistency from within.

**Model Collapse** (arXiv:2512.14879, 2024-2025): Self-consuming training loops exhibit exponential entropy decay without external coupling. Finite-sample noise forces systems to project onto ever-shrinking empirical support. Practical workarounds exist (real-data mixing, entropy bonuses, retrieval-augmented generation) but they all involve injecting external information — confirming that the system cannot bootstrap quality from within.

#### Active Research Frontier

The **ICLR 2026 Workshop on AI with Recursive Self-Improvement** confirms this is a major active research area. The workshop notes that frontier AI labs are automating large fractions of research and engineering operations, with "workforces" expected to grow from thousands to tens of thousands in 2026. The framing is explicitly about building "algorithmic foundations for powerful and reliable self-improving systems" — acknowledging both the potential and the unsolved challenges.

### Practical Verdict: What's Feasible and What Isn't

**Feasible — Bounded Configuration Optimization:**
The autoresearch harness has a finite set of meta-parameters that can be optimized with fixed external evaluation:
- **Prompt templates:** What instructions, context, and history to include. STOP and APO systems prove this works.
- **Acceptance criteria:** Greedy vs. annealing vs. threshold — the synergy A parameters.
- **Context allocation:** How much of the context window goes to code vs. history vs. diagnostics.
- **Evaluation protocol:** Training duration, number of seeds, warmup steps.
- **Iteration structure:** How often to run diagnostics, how often to update the bandit, etc.

These are bounded, enumerable dimensions. The evaluation metric (val_bpb on real data) is external and fixed. The meta-loop modifies harness configuration, not the evaluation itself. This satisfies the conditions that make STOP and AlphaEvolve work: fixed evaluator, bounded search space, external grounding.

**Not Feasible — Self-Modifying Evaluation:**
Letting the meta-loop modify the evaluation metric, the dataset, or the definition of success leads directly to the entropy decay proven in arXiv:2601.05280. The system would optimize its own yardstick, inevitably gaming it. This is the Goodhart's Law failure mode at the meta level — and unlike the base level (where Idea 2's learned evaluation harness is grounded by downstream benchmarks), there's no natural external anchor for "what makes a good research process."

**Not Feasible — Deep Recursion:**
More than one level of meta-optimization (meta-meta-autoresearch) is practically impossible. The signal becomes too noisy — "rate of improvement of rate of improvement" measured over 50-iteration windows has enormous variance, making it impossible to distinguish genuine meta-improvements from noise. RISE's finding that shallow recursion works but deep recursion shows diminishing returns applies here too.

### Recommended Approach

1. **Treat harness optimization as a bandit problem.** Define 5-10 configurable dimensions of the harness (prompt template variant, temperature, acceptance criterion type, context budget allocation, evaluation duration). Maintain posteriors over configurations using Thompson Sampling — the same framework as Synergy A, but at the meta level.

2. **Use STOP-style scaffolding.** The meta-loop generates harness configurations, runs the inner loop for K iterations, and scores by improvement rate. The LLM proposing configurations is fixed and external.

3. **Keep evaluation external and fixed.** Val_bpb on the real validation set is the ground truth at every level. Never let the meta-loop touch the evaluation pipeline.

4. **Run shallow.** One level of meta-optimization is enough. Optimize the harness, but don't optimize the meta-optimization process.

5. **Budget 10-20% of total compute for meta-exploration.** Run the inner loop with the best-known configuration 80% of the time, and with experimental configurations 20% of the time. This bounds the cost of meta-optimization while still exploring.

### Bottom Line

Meta-autoresearch is not "near impossible" — it's partially deployed in production today (APO, AlphaEvolve). What's impossible is *unbounded* recursive self-improvement without external grounding. The practical version — bounded meta-optimization of harness parameters with fixed external evaluation — is feasible, well-supported by existing research, and likely to yield real improvements over the human-designed defaults. The key is respecting the mathematical limits: fixed evaluator, bounded search space, shallow recursion, external grounding.
