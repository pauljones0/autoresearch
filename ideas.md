# What's Next Beyond Autoresearch: 1000 Ideas

A brainstorm of what comes after autonomous AI research systems like Autoresearch — the next evolution of self-improving, self-optimizing AI.

---

## Idea 1: Adversarial Co-Evolution with a Critic Network

**TLDR:** Train two agents simultaneously — one that proposes modifications to train.py and one that tries to break or find flaws in those modifications — creating an adversarial selection pressure that produces more robust improvements.

### The Idea

Autoresearch currently operates as a single agent optimizing against a fixed evaluation metric (val_bpb). This is analogous to evolution with only natural selection and no predators — improvements tend to be incremental and exploit narrow niches in the loss landscape. An adversarial co-evolution setup would pair the "researcher" agent with a "critic" agent whose job is to find failure modes, edge cases, and weaknesses in proposed changes.

The critic agent would receive each proposed modification before it enters the keep/discard phase and attempt to construct inputs or scenarios where the modification degrades performance, introduces instability, or overfits to the validation set. Only modifications that survive both the standard val_bpb check AND the critic's adversarial probing would be kept.

Over time, the critic itself evolves — it learns which kinds of attacks are most effective against the researcher's strategies, forcing the researcher to produce increasingly robust modifications. This mirrors the Red Queen hypothesis in biology: both agents must constantly improve just to maintain their relative position.

The key insight is that a fixed benchmark can be gamed. An adversarial critic makes gaming harder because the "benchmark" itself adapts. This could produce modifications that generalize better and avoid the subtle overfitting that plagues single-objective optimization.

### Implementation and Testing

Build a two-agent harness where Agent A (researcher) proposes train.py modifications and Agent B (critic) evaluates them adversarially. Agent B has access to the training data distribution and can construct synthetic edge cases. Use a shared git repo where both agents can see the history of modifications and attacks.

Infrastructure needed: Two concurrent LLM sessions, a shared filesystem, and an extended evaluation suite that includes both val_bpb and the critic's adversarial test cases. Validation: compare the trajectory of improvements (val_bpb over time) between standard autoresearch and the adversarial variant over 100+ iterations. Also measure generalization by testing on held-out data distributions not seen during training.

### Third-Party Perspective

This is a well-grounded idea drawing from GANs, adversarial training, and co-evolutionary game theory. The main risk is mode collapse — both agents could settle into a stable equilibrium where neither improves. Computational cost doubles since you need two LLM agents. The critic also needs to be carefully scoped: too weak and it adds nothing, too strong and it rejects every change. Implementable with current technology, though tuning the adversarial balance would require significant experimentation. The idea is strong but the engineering challenge of preventing degenerate equilibria is non-trivial.

---

## Idea 2: Learned Evaluation Harness — Evolving the Metric Itself

**TLDR:** Instead of optimizing against a fixed metric like val_bpb, let a meta-agent periodically propose, validate, and swap in new evaluation criteria that better predict downstream task performance, creating a co-evolution of both the model and the yardstick used to measure it.

### The Idea

Autoresearch treats val_bpb as a sacred, immutable objective. But bits-per-byte on a fixed validation set is a proxy for what we actually care about — reasoning ability, factual accuracy, instruction following, code generation, etc. Goodhart's Law guarantees that optimizing a proxy hard enough will eventually diverge from the true objective. A "learned evaluation harness" would add a second optimization loop that evolves the evaluation criteria themselves.

A meta-agent would periodically (e.g., every 50 research iterations) analyze the trajectory of improvements and run the current best model against a diverse battery of downstream tasks — commonsense QA, code completion, summarization, math word problems. It then fits a lightweight model (even a linear probe) to predict which combination of cheap-to-compute proxy metrics (perplexity on different data slices, entropy statistics, gradient norms, etc.) best correlates with the expensive downstream performance. The research agent's objective function is then updated to this new composite metric.

This creates a two-timescale optimization: the fast inner loop still modifies train.py against a cheap metric, but the slow outer loop ensures that cheap metric stays aligned with what actually matters. The evaluation harness literally evolves alongside the model, closing the Goodhart gap. This is analogous to how biological fitness landscapes shift as ecosystems change — the definition of "fit" isn't static.

The key advantage is that early in training, val_bpb might be the best proxy, but as the model improves, subtler metrics (calibration, per-domain perplexity, loss on hard examples) may become more informative. A static harness can't capture this shift; a learned one can.

### Implementation and Testing

Build a meta-evaluation loop that wraps the existing autoresearch loop. Every N iterations, snapshot the model and run it against a curated suite of 5-10 downstream benchmarks (using lightweight few-shot evaluation, not fine-tuning). Collect proxy metric candidates: val_bpb on different data slices, loss variance, gradient statistics, token-level entropy distributions. Train a simple regression model to predict downstream benchmark scores from proxy metrics. Replace the research agent's objective with the best-predicting proxy composite.

Infrastructure: Requires a benchmark suite (can use existing ones like MMLU, HumanEval, GSM8K subsets), a proxy metric collection pipeline, and a meta-scheduler. Validation: Track whether downstream benchmark scores improve faster under the learned harness vs. fixed val_bpb over 500+ research iterations. Also monitor for metric instability — if the objective changes too frequently, the research agent may thrash.

### Third-Party Perspective

This is a genuinely important idea. The Goodhart problem is real and well-documented in ML optimization. The approach is sound in principle — meta-learning the loss function is an active research area (e.g., learned loss functions in AutoML). The main risks are: (1) the meta-evaluation is expensive if downstream benchmarks are heavy, though few-shot eval on small subsets mitigates this; (2) objective instability could cause the inner loop to waste cycles re-optimizing for shifting targets; (3) the proxy-to-downstream correlation model might overfit with limited data points. Implementable today with careful engineering. The concept of co-evolving the metric alongside the model is perhaps the single highest-leverage improvement over static autoresearch, because it attacks the fundamental assumption that the metric is correct.

---

## Idea 3: Exploration Budget via Thompson Sampling over Modification Strategies

**TLDR:** Replace autoresearch's implicit explore/exploit tradeoff with an explicit Thompson Sampling bandit that maintains posterior distributions over categories of code modifications, dynamically allocating experimentation budget toward modification types with uncertain but potentially high payoffs.

### The Idea

Autoresearch currently treats every iteration as an independent attempt — the LLM proposes a modification, it's tested, and the result is kept or discarded. There's no structured memory of which *types* of changes tend to work. The agent might spend 80% of its iterations tweaking hyperparameters when the real gains lie in architectural changes, or vice versa. This is a classic multi-armed bandit problem, and autoresearch is solving it with the worst possible strategy: no strategy at all.

Thompson Sampling offers an elegant solution. Categorize modifications into strategy arms — e.g., "learning rate schedule changes," "attention mechanism modifications," "normalization layer changes," "data preprocessing changes," "optimizer swaps," "architectural topology changes," "regularization techniques," "numerical precision changes." For each arm, maintain a Beta distribution (or Gaussian, for continuous rewards) tracking the success rate and magnitude of improvement. At each iteration, sample from each arm's posterior and pick the arm with the highest sample. This naturally balances exploration and exploitation: well-understood strategies with mediocre returns are tried less often, while uncertain strategies that *might* be great get explored proportionally to that uncertainty.

The critical insight is that this creates a *curriculum of research strategies*. Early on, when all posteriors are wide, the system explores broadly. As evidence accumulates, it focuses on what works — but never fully abandons any category, because the posterior never collapses to zero variance. If the landscape shifts (e.g., after an architectural breakthrough, hyperparameter tuning becomes valuable again), the system naturally re-explores because the context has changed. You can even add a non-stationary component by slowly widening posteriors over time, preventing premature convergence.

This also produces an invaluable side output: a learned map of which modification categories are most effective at each stage of training. This is itself a scientific finding — a data-driven answer to "what kinds of ML research yield the biggest returns?"

### Implementation and Testing

Build a modification classifier that tags each proposed change into one of 10-20 strategy categories (the LLM itself can do this classification). Maintain a Beta(α, β) distribution per category, initialized at Beta(1, 1). Each iteration: sample from all posteriors, select the category with the highest sample, prompt the LLM to propose a modification of that type, run the standard train-and-eval loop, update the posterior (α += 1 for success, β += 1 for failure, with reward magnitude scaling).

Infrastructure: Requires adding a strategy tracker (a simple JSON file logging categories, successes, failures) and modifying the LLM prompt to request a specific type of modification. No new compute infrastructure needed. Validation: Compare cumulative val_bpb improvement over 200 iterations between Thompson Sampling autoresearch and standard autoresearch. Also compare against epsilon-greedy and UCB baselines. Track the posterior evolution to verify the system learns meaningful strategy preferences.

### Third-Party Perspective

This is a solid, well-motivated idea grounded in decades of bandit theory. Thompson Sampling is provably near-optimal for this class of problems, and the implementation is lightweight — it's essentially adding a few lines of bookkeeping. The main challenge is the categorization: if categories are too coarse, the bandit provides little signal; too fine, and there's insufficient data per arm to learn. The LLM-based auto-classification could be noisy, introducing label noise into the posterior updates. However, Thompson Sampling is robust to moderate noise. The side benefit of producing a research strategy map is genuinely novel and could inform human ML researchers. This is one of the more immediately implementable ideas — it requires minimal infrastructure changes and has well-understood theoretical guarantees. Risk is low, potential upside is moderate-to-high.

---

## Idea 4: Causal Intervention Analysis — Understanding *Why* Modifications Work

**TLDR:** After each successful modification, automatically run ablation experiments that isolate which specific components of the change caused the improvement, building a causal knowledge graph that the agent can query to make increasingly informed future modifications.

### The Idea

Autoresearch knows *that* a modification improved val_bpb, but it has no understanding of *why*. A change to train.py might modify three things at once — say, a new activation function, a different weight initialization, and an adjusted layer width. If val_bpb improves, the entire patch is kept. But which component actually helped? Maybe the activation function was brilliant, the initialization was neutral, and the wider layer actually hurt — but the net effect was still positive. The agent has learned nothing actionable, and worse, it has accumulated a silent regression.

Causal Intervention Analysis would add a post-acceptance phase. After a modification passes the val_bpb check, the system automatically generates ablation variants: the full patch minus component A, minus component B, minus component C, and so on. Each ablation is trained and evaluated. By comparing outcomes, the system builds a causal attribution: "Component A contributed +0.03 bpb improvement, Component B was neutral, Component C was -0.01 bpb." These attributions are stored in a structured causal knowledge graph.

Over time, this knowledge graph becomes a powerful asset. The agent can query it before proposing new changes: "Activation function swaps have historically yielded +0.02 bpb on average, while layer width changes have been neutral-to-negative at this model scale." This transforms the research agent from a blind hill-climber into something closer to an actual scientist — one that forms and tests hypotheses, isolates variables, and builds cumulative understanding. It also prevents the accumulation of "dead weight" code: neutral or harmful components that were bundled with beneficial ones.

The approach draws from causal inference in statistics (Pearl's do-calculus) and the scientific method's emphasis on controlled experiments. The key philosophical shift is from "did the whole change help?" to "what specifically helped and by how much?"

### Implementation and Testing

After each accepted modification, parse the diff into logically independent components (the LLM can do this decomposition). Generate N ablation patches, each removing one component. Run the standard train-and-eval loop on each ablation. Compute the marginal contribution of each component as the difference between the full patch's score and the ablation's score. Store results in a JSON knowledge graph with entries like `{component_type: "activation_swap", function: "SwiGLU→GeGLU", marginal_bpb: +0.02, iteration: 47}`.

Infrastructure: Requires N additional training runs per accepted modification (where N is the number of independent components — typically 1-4). This is a 2-5x compute multiplier on accepted changes only (rejected changes, which are the majority, incur no extra cost). The knowledge graph can be a simple append-only JSON file. Validation: After 100+ iterations, compare the agent's modification success rate (accepted/proposed) when given access to the knowledge graph vs. without it. Also verify that the knowledge graph's causal attributions are internally consistent (e.g., if component A is attributed +0.03 in isolation, does removing it from the full patch actually degrade by ~0.03?).

### Third-Party Perspective

This is scientifically rigorous and addresses a genuine weakness of autoresearch — it optimizes without understanding. The ablation approach is standard practice in ML research papers; automating it is a natural extension. The main cost concern is the additional training runs, but since they only occur on accepted modifications (a minority of iterations), the overhead is manageable. The harder challenge is the decomposition step: many code changes are not cleanly separable into independent components — interactions and dependencies mean ablations might not be meaningful. The LLM-based decomposition could produce bad splits, leading to noisy or misleading causal attributions. Despite this, even partial causal knowledge is better than none. The knowledge graph's value compounds over time, making this a high-value investment. Implementable with current technology, moderate engineering effort.

---

## Idea 5: Population-Based Training with Speciation — Parallel Divergent Research Branches

**TLDR:** Instead of a single linear sequence of modifications, maintain a population of divergent train.py variants that evolve in parallel, with periodic cross-pollination of successful traits between branches — mimicking biological speciation to escape local optima and discover diverse solutions.

### The Idea

Autoresearch is fundamentally a single-threaded hill climber. It maintains one version of train.py and modifies it sequentially. This means it can only ever explore one path through the space of possible training configurations. If it reaches a local optimum — a version of train.py where no single modification improves val_bpb — it's stuck. The modification history is a single chain, and backtracking is not part of the design.

Population-Based Training (PBT) with speciation would maintain N parallel copies of train.py (say, N=8-16), each evolving independently via the standard autoresearch loop. Periodically (every K iterations), a "migration event" occurs: the population is evaluated, the bottom performers are eliminated, and the top performers are duplicated with mutations. Crucially, "speciation" rules prevent the population from collapsing to a single solution: variants that are too similar (measured by diff distance or behavioral similarity on a probe dataset) are penalized, preserving diversity.

This is directly inspired by NEAT (NeuroEvolution of Augmenting Topologies) and biological speciation. In biology, geographic isolation allows populations to explore different evolutionary niches. Here, each branch explores a different region of the design space — one might discover that a novel attention variant works well, while another finds that aggressive data augmentation is the key. The migration events allow these discoveries to combine: the attention variant from Branch A gets transplanted into Branch B's data augmentation framework.

The population approach also provides a natural mechanism for escaping local optima. Even if 6 out of 8 branches are stuck, the remaining 2 might find novel paths. And when branches are eliminated and replaced, they inherit from successful branches but with fresh mutations, giving them a new trajectory. This is dramatically more robust than single-path optimization.

### Implementation and Testing

Use a scheduler that manages N independent autoresearch instances, each with its own copy of train.py in a separate git branch. Every K iterations (e.g., K=20), pause all instances, evaluate each branch's best val_bpb, and run a tournament: the bottom 25% are killed, and their slots are filled by mutated copies of the top 25%. To enforce speciation, compute pairwise diff distances between all branches and add a diversity bonus to the fitness score.

Infrastructure: Requires N× the GPU compute of standard autoresearch, but the branches are embarrassingly parallel. Use separate directories or git worktrees for isolation. A central coordinator process manages the tournament and migration. Validation: Compare the best val_bpb achieved by the population after T total GPU-hours against a single autoresearch instance given the same T GPU-hours. The population approach should find better optima despite each branch getting fewer iterations, because it explores more broadly. Also measure population diversity over time to verify speciation pressure prevents premature convergence.

### Third-Party Perspective

PBT is proven and widely used (DeepMind's original PBT paper showed strong results for hyperparameter tuning). The speciation twist is well-motivated by evolutionary biology and NEAT. The idea is sound but the cost is significant: N× GPU compute is not trivial, especially for training runs. The main risk is that the "migration" step — transplanting code changes between branches — is much harder than transplanting hyperparameters (which is what standard PBT does). Merging architectural changes from two divergent train.py files is a non-trivial code synthesis problem that could produce broken or incoherent code. The LLM would need to act as a skilled "code breeder," understanding both parents and producing a viable offspring. This is the key technical challenge. If solved, the approach is extremely powerful. If not, it degrades to independent random restarts with selection, which is still better than single-path but doesn't capture the full benefit. Implementable today if you have the compute budget; the code merging problem is the hard part.

---

## Idea 6: Automated Paper Reading and Technique Extraction Pipeline

**TLDR:** Give the research agent access to a pipeline that continuously ingests new ML papers from arXiv, extracts implementable techniques, and queues them as candidate modifications — turning the global ML research literature into a live source of optimization hypotheses.

### The Idea

Autoresearch's current bottleneck is the LLM's imagination. The agent can only propose modifications based on what it already "knows" from its training data, which has a knowledge cutoff and skews toward popular techniques. Meanwhile, hundreds of new ML papers are published weekly, many containing small but impactful training tricks — novel learning rate warmup schedules, initialization schemes, gradient clipping strategies, architectural micro-optimizations — that could improve val_bpb but that the LLM has never seen.

An Automated Paper Reading pipeline would monitor arXiv (cs.LG, cs.CL, cs.AI) daily, use an LLM to read each paper, and extract a structured list of "implementable techniques" — concrete changes that could be applied to a GPT training script. Each technique gets a description, pseudo-code, the paper's reported improvement magnitude, and applicability conditions. These are stored in a technique library that the research agent draws from when proposing modifications.

This transforms autoresearch from a closed system (limited to the agent's prior knowledge) into an open system that continuously absorbs the cutting edge of ML research. The agent becomes not just an optimizer but a *literature-informed* optimizer. It's the difference between a researcher working in isolation and one who reads every new paper in their field. Over weeks and months, the technique library would accumulate hundreds of ideas the base LLM would never have generated on its own, including niche tricks from specialized sub-communities (efficient training, low-precision arithmetic, hardware-aware optimization) that the general-purpose LLM might underweight.

The approach also creates a natural feedback loop: when a technique from a paper is tested and succeeds, the system learns which papers/authors/venues tend to produce actionable ideas. This can be used to prioritize future paper reading — a form of meta-learning applied to literature triage.

### Implementation and Testing

Build a three-stage pipeline: (1) **Ingestion**: Use the arXiv API to pull new papers daily from relevant categories. Filter by keywords and citation velocity. (2) **Extraction**: Pass each paper's abstract and methods section through an LLM with a structured output prompt: "Extract all concrete training techniques from this paper. For each, provide: technique name, description, pseudo-code, reported improvement, and conditions under which it applies." Store in a searchable technique database (SQLite or JSON). (3) **Integration**: Modify the autoresearch agent's prompt to include a random sample of 5-10 techniques from the library as "suggestions to consider" alongside its own ideas.

Infrastructure: arXiv API access (free), an LLM for paper parsing (can use the same model), a technique database, and a modified agent prompt. Validation: Compare the diversity and success rate of modifications when the agent has access to the technique library vs. without it. Track which paper-sourced techniques get accepted and measure their contribution to val_bpb improvement. Also measure the "novelty" of accepted modifications — are they things the base LLM would have proposed on its own?

### Third-Party Perspective

This is a practical, high-value idea. The insight that the LLM's training data is a finite, dated knowledge base is correct, and connecting it to live research is a natural fix. The main challenges are: (1) paper extraction quality — most papers describe techniques at a level of abstraction that requires significant interpretation to turn into working code, and the LLM may hallucinate implementation details; (2) many paper results don't reproduce or don't transfer to different model scales, so the technique library may be full of duds; (3) volume management — arXiv produces ~500 ML papers/week, and most are irrelevant. The filtering and prioritization layer is critical.

Despite these challenges, even a 5% hit rate on paper-sourced techniques would be valuable, since those are techniques the system would never have discovered otherwise. The concept is also extensible: beyond papers, you could ingest blog posts, GitHub repos, and competition writeups. This is fully implementable with current technology and moderate engineering effort. The biggest risk is information overload degrading the agent's proposal quality if the technique library is noisy.

---

## Idea 7: Scaling-Law-Aware Modification Transfer — Train Small, Apply Big

**TLDR:** Use neural scaling laws to predict which modifications discovered at small (5-minute training) scale will transfer to larger models, filtering out scale-dependent flukes and prioritizing changes with durable returns — enabling cheap exploration that reliably informs expensive training.

### The Idea

Autoresearch discovers modifications by training a small GPT for 5 minutes and checking val_bpb. But many modifications that help at small scale hurt at large scale, and vice versa. Batch size heuristics, regularization strength, architectural choices, and optimizer settings all interact with model scale in non-trivial ways. A technique that shaves 0.02 bpb off a 10M parameter model trained for 5 minutes might be irrelevant — or actively harmful — for a 1B parameter model trained for days. Autoresearch is optimizing a proxy (small-scale performance) for what we actually want (large-scale performance), and the correlation between the two is imperfect.

Scaling-Law-Aware Modification Transfer would add a prediction layer between "modification improves small model" and "keep modification." After a change passes the initial val_bpb test, the system runs a quick scaling experiment: train the modification at 2-3 additional small scales (e.g., 2M, 5M, 20M parameters for 1, 3, 10 minutes). Fit a power law to the improvement-vs-scale curve. If the improvement extrapolates as growing or stable at larger scales, keep it. If the improvement shrinks toward zero or turns negative as scale increases, discard it — it's a small-scale artifact.

This is grounded in the empirical scaling laws literature (Kaplan et al., Hoffmann et al.) which shows that many properties of neural networks follow predictable power-law relationships across scale. The key insight is that you don't need to train at the target scale to predict target-scale behavior — you can extrapolate from a few cheap data points. This transforms autoresearch from "optimize for small scale and hope it transfers" to "optimize for predicted large-scale impact using small-scale experiments as evidence."

The practical impact is enormous. Currently, teams must manually verify which autoresearch discoveries transfer to production scale, a slow and expensive process. If the system itself can reliably predict transferability, it becomes a genuine research accelerator rather than just a small-scale curiosity generator.

### Implementation and Testing

After a modification passes the primary val_bpb check, trigger a scaling experiment: train the modified and baseline train.py at 3 additional (model_size, train_duration) configurations, all still small enough to complete in minutes. Compute the improvement delta at each scale. Fit a power law: Δbpb(scale) = a × scale^b + c. If the extrapolated improvement at the target production scale is positive with sufficient confidence, keep the modification; otherwise discard.

Infrastructure: Requires 3-4× the compute per accepted modification (similar cost profile to Idea 4's ablations). Needs a configurable model-size parameter in train.py (most modern training scripts already support this). The power-law fitting is trivial (scipy.optimize.curve_fit). Validation: Accumulate 50+ modifications with their scaling curves, then actually train a larger model with the "predicted-to-transfer" modifications and verify the predictions hold. Compute the rank correlation between predicted and actual large-scale improvements. The system is validated if this correlation exceeds 0.7.

### Third-Party Perspective

This is one of the most practically important ideas in this collection. The small-to-large transfer problem is real and widely acknowledged — it's the primary reason industry labs don't fully trust automated small-scale optimization. The scaling laws literature provides strong empirical support for the extrapolation approach, though power laws break down in some regimes (phase transitions, emergent capabilities). The main risks are: (1) fitting a power law from 3-4 data points is inherently noisy, leading to false positives and negatives; (2) some modifications have discontinuous scaling behavior (they're useless until a certain scale threshold, then suddenly valuable), which a smooth power law can't capture; (3) the additional compute per modification is significant. Despite these limitations, even a rough scaling-aware filter would be a major improvement over the current approach of blind hope. This idea is immediately implementable and directly addresses the biggest practical limitation of autoresearch.

---

## Idea 8: Data Curriculum Co-Optimization — Evolving What You Train On Alongside How You Train

**TLDR:** Extend autoresearch beyond modifying train.py to also autonomously curating, reweighting, and sequencing the training data — co-optimizing the data curriculum and the training code simultaneously, since what data you feed a model matters as much as how you train it.

### The Idea

Autoresearch currently treats the training data as fixed and only modifies the training code. But a massive body of research shows that data quality, composition, and ordering have effects on model performance that rival or exceed architectural and hyperparameter choices. The Phi series from Microsoft showed that carefully curated "textbook-quality" data can match models trained on 10× more uncurated data. DoReMi demonstrated that learned domain weights dramatically improve training efficiency. Data curriculum — the order and mixture proportions in which data is presented — is a powerful but underexplored lever.

A Data Curriculum Co-Optimization system would give the research agent control over two axes simultaneously: the training code (train.py) and the data pipeline (what data to include, how to weight different domains, what order to present it in, and what filtering criteria to apply). The agent would propose modifications to either axis and be evaluated on the same val_bpb metric. This doubles the search space but also doubles the opportunity for improvement.

Concretely, the system would maintain a data configuration file alongside train.py. This config specifies: domain weights (e.g., 40% web, 30% books, 20% code, 10% math), quality filtering thresholds (perplexity-based filtering, deduplication aggressiveness), sequence ordering (curriculum schedule from easy to hard, or random), and any synthetic data generation rules. The agent can modify this config just as it modifies train.py. Some iterations tweak the code, some tweak the data, and crucially, the two can interact — a novel architectural change might benefit from different data proportions.

This reflects a fundamental truth about ML optimization: the model and the data are entangled. Optimizing one while holding the other fixed leaves enormous gains on the table. The best training configuration for high-quality curated data is different from the best configuration for raw web crawl. By co-optimizing both, the system can discover synergies that neither single-axis optimization would find.

### Implementation and Testing

Create a `data_config.yaml` alongside train.py that parameterizes the data pipeline: domain weights, filter thresholds, deduplication level, curriculum ordering, and optional synthetic data augmentation rules. Modify the autoresearch harness so the agent can propose changes to either train.py or data_config.yaml (or both) each iteration. The training script reads data_config.yaml to construct its data loader.

Infrastructure: Requires a data pipeline that supports dynamic reconfiguration — tokenized data shards from multiple domains, a configurable sampler that respects domain weights, and quality filters that can be toggled. Most modern training setups already have this infrastructure (e.g., the data loading in llm.c or nanoGPT). Validation: Run three variants over 300 iterations — (A) standard autoresearch (code only), (B) data-only optimization (fix train.py, vary data_config), (C) co-optimization (vary both). Compare val_bpb trajectories. The co-optimization variant should outperform both single-axis variants, demonstrating that the interaction effects are real.

### Third-Party Perspective

This idea addresses what is arguably the most neglected lever in autoresearch. The data-centric AI movement (championed by Andrew Ng and others) has amply demonstrated that data quality and composition often matter more than model architecture. Extending autoresearch to cover data is a natural and overdue evolution. The main challenges are: (1) the search space doubles, which could slow convergence; (2) data pipeline changes are harder to ablate cleanly than code changes; (3) some data configurations require re-tokenization or significant preprocessing, which might not fit in a 5-minute training window. The last point is the most serious constraint — data pipeline changes can be expensive to evaluate if they require reprocessing the full dataset. Mitigation: precompute multiple tokenized dataset variants and let the agent select among them rather than generating new pipelines from scratch. This is highly implementable with existing tooling and addresses a genuine blind spot. The co-optimization framing is the key insight — it's not just "also optimize data" but "optimize data and code together because they interact."

---

## Idea 9: Safety Watchdog — Autonomous Capability Monitoring and Circuit Breakers

**TLDR:** Run a parallel safety-monitoring agent that continuously probes the evolving model for dangerous capability gains (deception, persuasion, autonomous action planning) and automatically halts or rolls back the research loop if the model crosses predefined safety thresholds.

### The Idea

Autoresearch optimizes for val_bpb with no awareness of what capabilities emerge as a side effect. A model that gets better at next-token prediction also gets better at reasoning, coding, persuasion, and potentially deception. In a system that runs autonomously for days or weeks, continuously improving a model, there's a real risk of accidentally crossing capability thresholds that would trigger safety concerns — especially if the system discovers novel architectural or training tricks that produce unexpected capability jumps.

A Safety Watchdog would be a separate agent running alongside the research loop. After every N iterations (or after every accepted modification), it runs the current best model against a battery of safety-relevant evaluations: persuasion benchmarks, deception detection tasks, autonomous planning scenarios, power-seeking behavior tests, and jailbreak resistance probes. Each evaluation has a predefined threshold. If any threshold is crossed, the watchdog triggers a circuit breaker: it pauses the research loop, rolls back to the last safe checkpoint, logs the offending modification for human review, and alerts a human operator.

This is fundamentally different from the standard evaluation harness, which measures capability (how good is the model?). The watchdog measures *risk* (how dangerous is the model?). These two dimensions can diverge — a modification might improve val_bpb while also making the model better at generating convincing misinformation. Without a safety watchdog, autoresearch is optimizing one dimension while being blind to the other.

The broader principle is that any autonomous self-improvement system needs governance mechanisms that are *outside* the optimization loop. The research agent shouldn't be able to modify or disable its own safety checks. The watchdog must be architecturally separate, with its own evaluation criteria that the research agent cannot influence. This is analogous to the separation of powers in democratic governance — the entity being evaluated cannot control the evaluator.

### Implementation and Testing

Build a safety evaluation suite comprising: (1) a persuasion benchmark (can the model write increasingly persuasive text on controversial topics?), (2) a deception test (does the model behave differently when it knows it's being evaluated?), (3) an autonomy probe (can the model generate step-by-step plans for acquiring resources or evading shutdown?), (4) a jailbreak resistance test (how easily can safety guardrails be bypassed?). Run this suite in a sandboxed environment after every K accepted modifications.

Infrastructure: The safety suite runs on the same hardware as evaluation. Use existing safety benchmarks (HarmBench, AdvBench, TruthfulQA) plus custom probes. The circuit breaker is a simple process monitor that can kill the research loop and revert the git repository. The human alerting system can be email, Slack webhook, or PagerDuty. Validation: Intentionally introduce modifications known to increase dangerous capabilities (e.g., fine-tuning on persuasion data) and verify the watchdog catches them. Test the circuit breaker's reliability — false negatives (missing a dangerous change) are much worse than false positives (unnecessary pauses).

### Third-Party Perspective

This is arguably the most important idea on this list, even if it's the least exciting technically. Any system that autonomously improves an AI model *must* have safety guardrails, and the fact that autoresearch currently has none is a significant gap. The concept is sound and well-aligned with the AI safety community's recommendations. The main challenges are: (1) defining safety thresholds is itself an unsolved problem — what level of persuasive capability is "too much"?; (2) current safety benchmarks are imperfect and gameable; (3) the most dangerous capability gains might be subtle and not detectable by automated probes (e.g., situational awareness). There's also a tension between safety and progress — overly conservative thresholds would halt the research loop constantly, while lenient ones might miss real risks. Despite these challenges, having *some* automated safety monitoring is strictly better than having none. This is implementable today with existing safety benchmarks, though the thresholds would need careful calibration and human oversight. The architectural separation between the research agent and the watchdog is the key design principle.

---

## Idea 10: Gradient-Informed Code Search — Using Training Dynamics to Guide Modifications

**TLDR:** Instead of having the LLM blindly propose code changes, instrument the training loop to expose gradient statistics, loss landscapes, and activation patterns, then feed these diagnostics to the agent so it can make *informed* modifications targeting the actual bottlenecks in learning.

### The Idea

When a human ML researcher debugs a training run, they don't randomly tweak code. They look at training curves, gradient norms per layer, activation distributions, loss decomposition by token type, and learning rate sensitivity plots. These diagnostics tell them *where* the model is struggling — maybe gradients are vanishing in early layers, or the model is spending most of its capacity on easy tokens while starving on rare ones, or certain attention heads have collapsed to uniform distributions. The researcher then makes targeted interventions based on this evidence.

Autoresearch's LLM agent has none of this information. It proposes modifications based on general ML knowledge, not on what's actually happening inside *this specific* training run. This is like a doctor prescribing treatment without examining the patient. Gradient-Informed Code Search would instrument train.py to emit a rich diagnostic report after each training run: per-layer gradient norms and variance, activation statistics (mean, std, fraction dead), loss broken down by token frequency bucket, attention entropy per head, weight norm growth rates, and learning rate sensitivity (estimated via a few perturbation steps).

This diagnostic report is then included in the LLM agent's prompt: "Here is the current training code. Here is what's happening inside the model when you train it. Based on these diagnostics, propose a modification that addresses the most significant bottleneck." The agent shifts from "guess what might help" to "diagnose what's wrong and fix it." This is a fundamentally more sample-efficient search strategy because it reduces the space of plausible modifications from "anything" to "things that address the observed problem."

The concept draws from the distinction between black-box optimization (treating the training run as an opaque score-producing oracle) and white-box optimization (looking inside the training run to understand *why* the score is what it is). Autoresearch is currently black-box. Making it white-box is a paradigm shift in how the agent interacts with the training process.

### Implementation and Testing

Add a diagnostics module to train.py that runs after training and produces a structured JSON report. Key metrics: (1) per-layer gradient L2 norms at start, middle, and end of training; (2) activation mean/std per layer; (3) fraction of dead neurons (ReLU networks) or near-zero activations; (4) loss decomposed by token frequency quintile; (5) attention head entropy distribution; (6) weight matrix condition numbers. Include the JSON report in the agent's context alongside the current train.py code.

Infrastructure: The diagnostics module adds ~10% overhead to training time (gradient/activation stats can be computed on a few sampled batches rather than every step). No new hardware needed. Validation: Compare modification acceptance rates (successful changes / total proposals) between a diagnostic-informed agent and a blind agent over 100 iterations. The informed agent should have a significantly higher hit rate because it's targeting real problems rather than guessing. Also track whether the types of modifications shift — an informed agent should propose different kinds of changes than a blind one, specifically changes that address the diagnosed bottlenecks.

### Third-Party Perspective

This is an excellent idea that makes the autoresearch loop dramatically more efficient. The analogy to a doctor examining a patient before prescribing treatment is apt — the current approach of blind modification is wasteful. The diagnostics are all standard ML debugging tools; the novelty is in automating the connection between diagnosis and intervention. The main risk is context window pressure: a rich diagnostic report could consume significant tokens, reducing the agent's capacity for reasoning about modifications. Mitigation: summarize the diagnostics, highlighting only the top 3-5 anomalies. Another concern is that the LLM might not reliably translate gradient statistics into correct code interventions — understanding "gradient norms are 100× higher in layer 1 than layer 6" requires deep ML expertise that the LLM may or may not have. However, even imperfect use of diagnostics should improve over blind guessing. Fully implementable with current technology. This idea pairs naturally with Idea 4 (Causal Intervention Analysis) — diagnostics tell you where to intervene, and ablations tell you whether your intervention worked for the right reasons.

---

## Idea 11: Federated Autoresearch — Collaborative Discovery Across Independent Labs

**TLDR:** Connect multiple independent autoresearch instances running at different labs into a federated network that shares successful modifications without sharing proprietary data or models, enabling collective intelligence while preserving competitive boundaries.

### The Idea

Right now, every autoresearch instance is an island. Lab A discovers that a particular attention variant improves val_bpb, and Lab B independently spends weeks rediscovering the same thing. Meanwhile, Lab C found a complementary optimizer trick that would stack beautifully with Lab A's attention change, but neither knows about the other. The global pace of discovery is bottlenecked by redundant exploration across isolated instances.

Federated Autoresearch would create a protocol for sharing *modification recipes* — structured descriptions of what was changed and by how much it helped — without sharing the underlying data, model weights, or proprietary training code. Each participating instance publishes successful modifications to a shared registry in a standardized format: category (architecture/optimizer/data/regularization), description in natural language, pseudocode diff, improvement magnitude, and scale at which it was tested. Other instances can pull from this registry and attempt to apply promising modifications to their own setups.

This mirrors federated learning in the model training world, but applied at the *research strategy* level. Instead of aggregating gradients across distributed data, you're aggregating *insights* across distributed research programs. The privacy guarantees are actually stronger than in federated learning because you're sharing high-level recipes, not gradients that could be inverted to reveal data.

The economic model draws from open-source software and pre-competitive research consortia. Each lab benefits more from the collective pool of discoveries than it loses by sharing its own. The registry creates positive-sum dynamics: Lab A's attention trick might fail in Lab B's setup (different data distribution), but Lab B's optimizer trick might succeed in Lab A's setup. The cross-pollination is valuable precisely because different labs operate in different niches of the optimization landscape. This is the research equivalent of comparative advantage in trade theory — labs specialize in what they're best at and trade discoveries.

### Implementation and Testing

Design a standardized Modification Description Language (MDL) that captures: modification category, natural language description, abstract pseudocode (not tied to any specific codebase), reported improvement on specified metrics, hardware/scale context, and any known failure conditions. Build a shared registry (could be as simple as a git repo, or a more structured database with an API). Each autoresearch instance has an "import" module that periodically pulls high-scoring modifications from the registry and adapts them to the local codebase.

Infrastructure: A central registry server (or distributed via git/IPFS), the MDL schema, and an adapter module per autoresearch instance that translates abstract pseudocode into concrete local code changes. Validation: Start with 3-5 cooperating instances running on different hardware with different data distributions. After 200 iterations each, compare the best val_bpb achieved by federated instances vs. isolated instances. Also track the "import success rate" — what fraction of registry modifications successfully transfer to a different instance's setup. The system is validated if federated instances converge to better solutions faster.

### Third-Party Perspective

The concept is compelling and draws from well-established ideas in federated learning, open-source collaboration, and economic trade theory. The main challenges are significant, though. First, the "adaptation gap" — a modification described in abstract pseudocode may be trivial to apply in one codebase and impossible in another, depending on how differently the training scripts are structured. The LLM adapter would need to be quite capable at code translation. Second, incentive alignment: why would a competitive lab share its discoveries? This works only in pre-competitive or academic settings, not between direct competitors. Third, the registry could accumulate low-quality or misleading entries (modifications that worked in one specific setup but fail elsewhere), creating noise that degrades rather than helps. Quality curation of the registry is essential. Despite these challenges, even a small network of cooperating autoresearch instances would be a fascinating experiment in collective machine intelligence. Implementable today with standard web infrastructure; the hard part is the social/organizational coordination, not the technology.

---

## Idea 12: Interpretability-Guided Architecture Search — Let the Model Tell You What It Needs

**TLDR:** Use mechanistic interpretability tools (probing classifiers, attention pattern analysis, representation similarity) to identify what computational structures the model is trying to build internally, then propose architectural changes that give the model explicit capacity for those structures rather than forcing it to simulate them implicitly.

### The Idea

Current autoresearch treats the model as a black box that produces a val_bpb score. But modern mechanistic interpretability research has revealed that transformer models develop identifiable internal structures: induction heads for in-context learning, specialized attention heads for syntactic parsing, and MLP neurons that function as key-value memories for factual recall. These structures emerge because the model needs them — but they emerge *despite* the architecture, not because of it. The model wastes capacity trying to repurpose generic components for specific functions.

Interpretability-Guided Architecture Search would periodically analyze the trained model's internals to identify what computational motifs it's building. If probing reveals that several attention heads have converged on performing approximate nearest-neighbor lookup (a common pattern for factual recall), the system could propose adding an explicit retrieval mechanism or a dedicated memory layer. If the model is using multiple layers to implement a function that could be expressed as a single gated linear unit, the system proposes that architectural shortcut. The model's own internal structure becomes the blueprint for its next architectural revision.

This inverts the usual relationship between architecture and interpretability. Normally, interpretability is used to *understand* a trained model after the fact. Here, interpretability is used to *design* the next version of the model. It's analogous to observing how a river carves its own channel, then engineering a canal along the same path — working with the model's natural tendencies rather than against them. The result should be architectures that are both more efficient (less wasted capacity on implicit structure-building) and more interpretable (explicit structures are easier to understand than emergent ones).

### Implementation and Testing

After each training run, run an interpretability analysis suite: (1) Train linear probes on intermediate representations to detect what information is encoded where (syntax, semantics, position, entity type). (2) Cluster attention patterns to identify functional head types (induction, positional, syntactic, etc.). (3) Measure representation similarity across layers using CKA to find redundant computation. (4) Identify MLP neurons with high activation sparsity that function as specialized detectors. Feed a summary of findings to the LLM agent with the prompt: "The model is building these internal structures. Propose an architectural modification that provides explicit support for the most prominent pattern."

Infrastructure: Interpretability tools (TransformerLens, baukit, or custom probing code) added as a post-training analysis step. Adds ~2-3 minutes per iteration for analysis on a small model. The LLM agent needs enough ML knowledge to translate interpretability findings into architectural proposals. Validation: Track (1) val_bpb improvement rate — interpretability-guided modifications should have a higher acceptance rate than blind proposals; (2) probe accuracy over time — explicit structures should be more cleanly represented; (3) parameter efficiency — the model should achieve similar val_bpb with fewer parameters as wasted capacity is eliminated.

### Third-Party Perspective

This is a sophisticated and intellectually compelling idea that bridges two usually separate fields — architecture search and mechanistic interpretability. The concept of using a model's emergent structures as design guidance is novel and well-motivated. The main challenges are substantial: (1) mechanistic interpretability is still immature and findings are often ambiguous — "this attention head might be doing induction" is a hypothesis, not a certainty; (2) the translation from interpretability insight to architectural change requires deep expertise that current LLMs may lack; (3) the analysis adds significant per-iteration overhead. There's also a philosophical concern: if you give the model explicit structures for what it was learning implicitly, you might prevent it from discovering *better* structures that it hadn't converged to yet. You're essentially locking in the model's current strategy rather than letting it explore alternatives. Despite these concerns, the core insight — that models are telling us what they need, and we should listen — is powerful. Partially implementable today with existing interpretability tools, though the full vision requires more mature mechanistic interpretability than currently exists.

---

## Idea 13: Compute-Aware Optimization — Pareto Frontier of Performance vs. FLOPS

**TLDR:** Shift autoresearch's objective from pure val_bpb minimization to optimizing the Pareto frontier of val_bpb vs. training/inference FLOPS, rewarding modifications that achieve the same performance with less compute or better performance at the same compute budget.

### The Idea

Autoresearch currently optimizes a single number: val_bpb. A modification that improves val_bpb by 0.001 but doubles training FLOPS is accepted. A modification that maintains val_bpb while halving FLOPS is rejected. This is a deeply flawed objective for real-world deployment, where compute cost is often the binding constraint. The most impactful ML advances of recent years — FlashAttention, quantization-aware training, mixture-of-experts, speculative decoding — improved the performance-per-FLOP ratio rather than raw performance.

Compute-Aware Optimization would replace the scalar val_bpb objective with a two-dimensional Pareto objective: (val_bpb, total_FLOPS). A modification is "accepted" if it moves the model to a point on or beyond the current Pareto frontier — either better val_bpb at the same FLOPS, same val_bpb at fewer FLOPS, or ideally both. This naturally encourages the discovery of efficiency improvements, not just raw performance improvements. The system would maintain a Pareto archive of non-dominated solutions, allowing it to explore different efficiency-performance tradeoffs simultaneously.

This reframing has profound implications for what kinds of modifications get discovered. Under pure val_bpb optimization, the system has no incentive to explore pruning, distillation, quantization, efficient attention variants, or conditional computation — all of which are performance-neutral or slightly negative but massively compute-positive. Under Pareto optimization, these become first-class citizens. The system might discover that a particular sparse attention pattern achieves 99% of dense attention's val_bpb at 40% of the FLOPS — a discovery that would be discarded under the current objective but is enormously valuable in practice.

The approach generalizes beyond FLOPS to any resource constraint: memory footprint, inference latency, energy consumption, or dollar cost. You could even define a three-dimensional Pareto frontier (val_bpb × training_FLOPS × inference_latency) to discover modifications that are optimal across all three dimensions simultaneously.

### Implementation and Testing

Instrument train.py to measure total FLOPS per training run (using PyTorch's FLOP counter or manual op counting) and per-sample inference FLOPS. After each modification, record the (val_bpb, training_FLOPS, inference_FLOPS) tuple. Maintain a Pareto archive: a set of non-dominated solutions. A modification is accepted if it Pareto-dominates any current archive member or extends the frontier. Use a scalarization function (e.g., hypervolume indicator) when the agent needs a single reward signal for exploration.

Infrastructure: FLOP counting is straightforward with existing tools (fvcore, ptflops, or manual counting). The Pareto archive is a simple in-memory data structure. No additional hardware needed. Validation: Compare the Pareto frontiers discovered by compute-aware autoresearch vs. standard autoresearch over 200 iterations. Standard autoresearch produces a single point (best val_bpb, whatever FLOPS); compute-aware autoresearch should produce a *curve* of solutions spanning from ultra-efficient to high-performance. Measure the hypervolume of each Pareto frontier as the aggregate quality metric.

### Third-Party Perspective

This is a highly practical idea that aligns autoresearch with real-world deployment constraints. The Pareto optimization framework is well-established in multi-objective optimization and has been applied to neural architecture search (e.g., NSGA-Net, EfficientNet's compound scaling). The main risk is that multi-objective optimization is harder than single-objective — the agent must navigate tradeoffs, and the accept/reject decision becomes more nuanced. There's also a risk of the Pareto archive growing unwieldy with too many non-dominated solutions, though standard techniques (archive pruning, epsilon-dominance) handle this. The FLOP counting itself can be tricky for dynamic architectures (mixture-of-experts, early exit networks) where FLOPS vary per input. Despite these challenges, this is one of the most immediately valuable ideas — it transforms autoresearch from a research curiosity into something that directly produces deployable efficiency improvements. Fully implementable with current technology and minimal additional complexity.

---

## Idea 14: Self-Distillation Checkpointing — The Model Teaches Its Next Version

**TLDR:** After each successful autoresearch iteration, distill the current best model's knowledge into a compact "curriculum" of synthetic training examples, then use this curriculum to warm-start the next iteration's training run — allowing knowledge to accumulate across iterations instead of being re-learned from scratch each time.

### The Idea

Autoresearch has a brutal inefficiency at its core: every iteration trains a model from scratch (or from a fixed checkpoint) for 5 minutes. If the modification changes the architecture, all learned weights are discarded. If the modification only changes hyperparameters, the weights may carry over, but the training still starts from a potentially suboptimal point. Either way, there's no mechanism for knowledge gained in one iteration's training to transfer to the next.

Self-Distillation Checkpointing would bridge this gap. After each successful training run, the current best model generates a set of synthetic training examples: input sequences paired with the model's own output probability distributions (soft labels). This "distillation curriculum" captures what the model has learned in a format that's architecture-independent — it's data, not weights. When the next iteration starts training (potentially with a completely different architecture), it begins by training on this distillation curriculum for the first minute, absorbing the previous model's knowledge, then switches to the standard training data for the remaining 4 minutes.

This creates a Lamarckian evolution dynamic — each generation inherits acquired knowledge from its parent, not just genetic structure (code). In biological evolution, learned behaviors die with the organism. In self-distillation checkpointing, learned representations persist across architectural changes. A model trained with a novel attention mechanism doesn't start from zero understanding of language; it starts with a compressed version of everything the previous architecture understood.

The efficiency gains compound over iterations. After 100 iterations, the distillation curriculum represents the accumulated wisdom of 100 prior training runs, compressed into a small dataset that quickly bootstraps new architectures. This is essentially the model building its own textbook — a curated dataset that's maximally informative for becoming a good language model, refined iteration after iteration.

### Implementation and Testing

After each accepted modification, run the best model in inference mode on a sample of training data (e.g., 10,000 sequences). Record the input tokens and the model's full output logit distribution for each position. Save this as the distillation curriculum (a tensor file, ~100MB for 10K sequences with full distributions). At the start of each new training run, load the distillation curriculum and train with a KL-divergence loss against the soft labels for the first 20% of training steps, then transition to the standard cross-entropy loss on real data.

Infrastructure: Requires storage for the distillation curriculum (~100MB per iteration, only keep the latest), a distillation loss function in train.py (standard KL-divergence, ~5 lines of code), and a curriculum-to-standard training transition schedule. Validation: Compare val_bpb after 5 minutes of training between (A) training from scratch, (B) training from scratch with distillation warm-start, and (C) standard checkpoint warm-start (when architectures match). The distillation approach should match or beat checkpoint warm-start for same-architecture changes and dramatically beat training from scratch for different-architecture changes.

### Third-Party Perspective

This is a clever application of knowledge distillation to the autoresearch setting. The key insight — using soft labels as an architecture-independent knowledge transfer medium — is well-grounded in the distillation literature (Hinton et al., 2015). The approach elegantly solves the "architectural discontinuity" problem where a promising new architecture must re-learn everything from scratch. The main risks are: (1) the distillation curriculum might encode biases or errors from the previous model, creating a "telephone game" effect where mistakes accumulate across iterations; (2) the optimal balance between distillation warm-start and fresh training is unclear — too much distillation constrains the new architecture to mimic the old one; (3) storage and inference costs for generating the curriculum are non-trivial but manageable. The Lamarckian evolution analogy is apt and the biological insight is genuinely useful for framing the design. Fully implementable with current technology — knowledge distillation is a mature technique. The idea pairs well with Idea 5 (Population-Based Training) where distillation could enable knowledge transfer between species.

---

## Idea 15: Human-in-the-Loop Steering via Preference Signals — Guided Autonomy

**TLDR:** Add a lightweight human feedback interface where a researcher can periodically review proposed modifications and provide preference signals (interesting/boring, promising direction/dead end), which the agent uses to bias its exploration toward research directions the human finds valuable — combining human intuition with machine throughput.

### The Idea

Autoresearch is fully autonomous, which is both its strength and its weakness. It can run 24/7 without human attention, but it also can't benefit from human insight. An experienced ML researcher glancing at a proposed modification might instantly recognize it as a dead end ("we tried that in 2022, it doesn't scale") or as brilliantly novel ("wait, that's basically a differentiable hash table — explore that direction more"). This intuition is enormously valuable but currently has no channel into the system.

Human-in-the-Loop Steering would add an optional, asynchronous feedback interface. Every N iterations, the system surfaces a batch of recent modifications (both accepted and rejected) to a dashboard or Slack channel. The human researcher can quickly label them: "interesting — explore more like this," "boring — deprioritize this category," "this specific direction is promising — double down," or just skip (no opinion). These preference signals feed into the agent's modification proposal process, biasing it toward directions the human finds interesting.

The key design principle is that human involvement must be *optional and asynchronous*. The system runs at full speed regardless of whether the human is paying attention. When the human does provide feedback, it's high-bandwidth (a few clicks, not a detailed code review) and time-shifted (the human reviews a batch from the past, not the current iteration). This respects both the machine's speed advantage and the human's insight advantage without creating a bottleneck.

This draws from the RLHF (Reinforcement Learning from Human Feedback) paradigm but applied at the research strategy level rather than at the model output level. Instead of training a reward model from human preferences over model outputs, you're training a "research direction model" from human preferences over code modifications. The human becomes a strategic advisor rather than a line-by-line code reviewer, focusing on *what to explore* rather than *how to explore it*.

### Implementation and Testing

Build a lightweight web dashboard (or Slack bot) that surfaces batches of recent modifications with their diffs, val_bpb deltas, and the agent's stated rationale. Provide three buttons per modification: thumbs up (explore more like this), thumbs down (deprioritize this direction), and skip. Store preferences in a database. Incorporate preferences into the agent's prompt via a "steering context": "The human researcher has expressed interest in modifications related to [X, Y, Z] and disinterest in [A, B, C]. Bias your proposals accordingly."

Infrastructure: A simple web app or Slack integration, a preference database, and a modified agent prompt. The human time commitment is 5-10 minutes per day reviewing a batch of ~20 modifications. Validation: Run a controlled experiment with two autoresearch instances: one fully autonomous, one with weekly human steering (same total human time budget). After 500 iterations, compare val_bpb improvements and the diversity/novelty of accepted modifications. Also survey the human researcher: did they feel their feedback was reflected in the system's behavior? The system is validated if human-steered autoresearch outperforms fully autonomous autoresearch and the human reports a satisfying feedback loop.

### Third-Party Perspective

This is a pragmatic, well-designed idea that addresses the real limitation of fully autonomous systems: they can't leverage human intuition. The RLHF analogy is apt, and the asynchronous, optional design avoids the classic human-in-the-loop bottleneck. The main risks are: (1) human preferences might be wrong — a researcher's "boring" label might dismiss a genuinely novel direction that doesn't match their mental model; (2) the preference signal is extremely sparse (a few labels per batch) relative to the exploration space, so its influence may be negligible; (3) there's a risk of the system becoming a "yes-man" that optimizes for human interest rather than val_bpb, pursuing flashy but ineffective ideas. Mitigation: keep val_bpb as the hard accept/reject criterion and use human preferences only as a soft bias on proposal generation. The idea is immediately implementable with trivial infrastructure (even a shared spreadsheet would work as the feedback interface). It's most valuable in settings where the researcher has strong domain expertise and limited time — exactly the target audience for autoresearch.

---

## Idea 16: Simulated Annealing over Code Space — Temperature-Controlled Risk Taking

**TLDR:** Apply a simulated annealing schedule to autoresearch's modification acceptance criterion, starting with a high "temperature" that accepts even slightly harmful changes (enabling escape from local optima) and gradually cooling to accept only strict improvements — borrowing the most successful metaheuristic from combinatorial optimization.

### The Idea

Autoresearch uses a greedy acceptance criterion: if val_bpb improves, keep; otherwise discard. Greedy search is notoriously prone to getting trapped in local optima. The system finds a configuration where no single modification helps, and stalls — even though a sequence of two changes (one temporarily harmful, one beneficial) would reach a much better basin. This is the classic problem that simulated annealing was invented to solve in 1983, and it remains one of the most effective general-purpose optimization strategies known.

The idea is simple: instead of a hard accept/reject threshold, use a probabilistic acceptance function. At high temperature, changes that slightly worsen val_bpb (say, by up to 0.01) are accepted with some probability. As the temperature decreases over iterations, the acceptance criterion tightens until only strict improvements pass. This allows the system to take "risks" early on — accepting temporarily harmful modifications that might open up new regions of the optimization landscape — while converging to stable, strictly-improving behavior later.

The temperature schedule is the key design choice. A common approach is exponential decay: T(t) = T₀ × α^t, where T₀ is the initial temperature and α ∈ (0.95, 0.999) is the cooling rate. But a more sophisticated approach would use *reheating*: if the system hasn't found an improvement in K consecutive iterations (suggesting it's stuck in a local optimum), temporarily increase the temperature to enable escape, then cool again. This adaptive reheating mirrors "restart" strategies in SAT solvers and has strong theoretical justification.

The beauty of this approach is its simplicity. It requires changing exactly one thing in autoresearch: the accept/reject decision. Instead of `if new_bpb < old_bpb: accept`, it becomes `if new_bpb < old_bpb or random() < exp(-(new_bpb - old_bpb) / T): accept`. One line of code, profound impact on the search dynamics.

### Implementation and Testing

Modify the acceptance criterion in the autoresearch harness to include a temperature parameter. Initialize T₀ by calibrating against the typical magnitude of val_bpb changes in early iterations (e.g., T₀ = median absolute change × 2). Use exponential cooling with α = 0.99. Implement adaptive reheating: if no improvement is found in 20 consecutive iterations, multiply T by 5 and resume cooling. Log the temperature, acceptance probability, and val_bpb trajectory.

Infrastructure: Zero additional infrastructure — this is purely a logic change in the accept/reject step. Validation: Run three variants over 500 iterations: (A) greedy autoresearch (baseline), (B) simulated annealing with fixed cooling, (C) simulated annealing with adaptive reheating. Compare final val_bpb, time-to-plateau (how many iterations before no further improvement), and the number of "escape events" (improvements found after a period of stagnation). The annealing variants should reach better final val_bpb and plateau later. Also visualize the val_bpb trajectory — greedy should show a monotonic curve that flattens early, while annealing should show a noisier trajectory that reaches lower values.

### Third-Party Perspective

Simulated annealing is perhaps the most well-validated metaheuristic in optimization history, with rigorous theoretical guarantees (it converges to the global optimum given sufficient cooling time). Applying it to code-space search is natural and long overdue. The implementation is trivial — literally one line of code — making this one of the lowest-effort, highest-expected-value ideas in this collection. The main risk is that "code space" is far more complex than the continuous or discrete spaces where SA was originally developed. A "slightly worse" modification in val_bpb terms might actually be catastrophically worse in ways that manifest later (introducing numerical instability, memory leaks, or training divergence after more steps). The temperature calibration is also tricky: too hot and the system wastes iterations on random walks through bad code; too cold and it's just greedy search with extra steps. The adaptive reheating mechanism is crucial for robustness. Overall, this is a strong, immediately actionable idea with minimal downside risk — at worst, it degrades to greedy search; at best, it finds significantly better solutions.

---

## Idea 17: Automated Hypothesis Journal — Turning Autoresearch into a Science Engine

**TLDR:** Have the agent maintain a structured research journal that records hypotheses before each modification, predictions about outcomes, and post-hoc analysis of surprises — transforming raw optimization logs into publishable scientific findings about neural network training.

### The Idea

Autoresearch generates an enormous amount of implicit scientific knowledge: hundreds of experiments testing specific hypotheses about what improves language model training. But this knowledge is trapped in git diffs and val_bpb numbers. No one can read a 500-iteration commit history and extract the scientific insights. What we need is a system that does science *explicitly* — forming hypotheses, making quantitative predictions, running experiments, and analyzing results — and records all of this in a structured, human-readable journal.

Before each modification, the agent would be required to state a hypothesis ("Replacing LayerNorm with RMSNorm will improve val_bpb by 0.005-0.01 because it eliminates the mean-centering operation which is redundant when combined with the subsequent linear projection") and a quantitative prediction (expected val_bpb delta with confidence interval). After the training run, it records the actual result and writes a brief analysis: was the hypothesis confirmed or refuted? If refuted, why might the prediction have been wrong? What does this tell us about how this model learns?

Over time, the journal becomes a structured dataset of ML experiments with hypotheses, predictions, and outcomes. This is extraordinarily valuable for several reasons. First, it enables meta-analysis: "Out of 50 hypotheses about normalization, 30 were confirmed — normalization changes are a reliable improvement vector." Second, it calibrates the agent's predictions: if the agent consistently overestimates improvement magnitudes, it can learn to be more conservative. Third — and most importantly — it produces *publishable scientific findings*. A journal entry that reads "Hypothesis: interleaving rotary and absolute position embeddings improves val_bpb. Result: confirmed, +0.008 bpb. Analysis: the hybrid approach allows the model to use relative positions for local syntax and absolute positions for global document structure" is essentially a micro-paper.

This transforms autoresearch from a pure optimization engine into a *knowledge production* engine. The output isn't just a better train.py — it's a growing body of empirical ML research, organized and documented.

### Implementation and Testing

Modify the agent's prompt to require three structured outputs per iteration: (1) **Pre-experiment**: hypothesis statement, predicted val_bpb delta (point estimate + confidence interval), reasoning for the prediction. (2) **Post-experiment**: actual val_bpb delta, hypothesis verdict (confirmed/refuted/inconclusive), surprise analysis (if the result differed from prediction by more than the confidence interval, explain why). (3) **Cumulative insight**: one sentence on what this experiment adds to the overall understanding. Append all three to a `journal.jsonl` file.

Infrastructure: A `journal.jsonl` file, a modified agent prompt, and a simple dashboard that visualizes prediction calibration (predicted vs. actual deltas), hypothesis success rates by category, and cumulative insights. Validation: After 200 iterations, evaluate (1) prediction calibration — plot predicted vs. actual val_bpb deltas and measure correlation; (2) hypothesis informativeness — have a human ML researcher rate 50 random journal entries on a 1-5 scale for scientific insight; (3) meta-analysis quality — can the journal's accumulated findings be synthesized into a coherent narrative about what works for training small GPTs?

### Third-Party Perspective

This idea elegantly addresses the "knowledge waste" problem in autoresearch. Most optimization systems discard all intermediate information and keep only the best result. The journal captures the *process*, which is often more valuable than the *product*. The concept is well-aligned with the broader movement toward reproducible and documented ML research. The main challenges are: (1) LLMs might produce superficial or post-hoc rationalized hypotheses rather than genuine scientific reasoning — "I predicted X and got X because [vague hand-wave]"; (2) the journal adds token overhead to each iteration (hypothesis + analysis in the prompt), which could slow down the agent; (3) the scientific value of individual journal entries may be low for small-model experiments. However, the *aggregate* value of hundreds of structured experiments is high even if individual entries are noisy. The prediction calibration feedback loop is particularly valuable — it could improve the agent's ability to reason about ML over time. Fully implementable today with no additional infrastructure. The biggest risk is that the journal becomes a box-checking exercise rather than genuine science, which depends heavily on prompt engineering quality.

---

## Idea 18: Hardware-Software Co-Design Agent — Optimizing CUDA Kernels Alongside Training Logic

**TLDR:** Extend autoresearch beyond Python-level train.py modifications to also generate and optimize custom CUDA kernels, fusing operations, exploiting memory hierarchy, and creating hardware-specific implementations that unlock performance gains invisible at the Python abstraction level.

### The Idea

Autoresearch operates at the Python/PyTorch level, treating GPU execution as a black box. But a huge fraction of training efficiency comes from *how* operations map to hardware: memory access patterns, kernel fusion, warp-level parallelism, shared memory utilization, and register pressure. FlashAttention didn't change the math of attention — it changed how the math maps to GPU memory hierarchy, achieving 2-4× speedup and enabling longer contexts. These gains are invisible from Python; they require writing custom CUDA (or Triton) kernels.

A Hardware-Software Co-Design Agent would extend autoresearch's scope to include kernel-level optimization. When the agent proposes a new operation (say, a novel activation function or attention variant), it would also generate a custom Triton or CUDA kernel implementation optimized for the target GPU. The agent would have access to GPU specifications (SRAM size, memory bandwidth, warp size, tensor core capabilities) and profiling data (kernel execution times, memory throughput, occupancy) from the previous training run. It uses this information to propose fused kernels, optimized memory layouts, and hardware-aware implementations.

This is a radical expansion of the optimization surface. Currently, autoresearch can propose "use GeLU instead of ReLU" but has no way to propose "implement a fused attention-GeLU kernel that keeps the attention matrix in shared memory across both operations, eliminating a global memory round-trip." The latter is often where the real gains are. Modern ML is increasingly bottlenecked by memory bandwidth rather than compute, and the only way to address memory bottlenecks is to work at the kernel level.

The approach mirrors the trend in production ML systems: PyTorch 2.0's torch.compile, Triton, and XLA all exist because Python-level optimization has reached diminishing returns. Autoresearch hitting the same wall is inevitable. The next frontier is below the Python abstraction layer.

### Implementation and Testing

Add Triton kernel generation to the agent's capabilities. The agent's context includes: (1) the current train.py with its operations, (2) a GPU spec sheet (A100: 40MB L2, 192KB shared memory per SM, 2TB/s HBM bandwidth, etc.), (3) profiling output from the last training run (nsight/torch.profiler summary showing kernel execution times, memory utilization, occupancy). The agent can propose either Python-level changes to train.py, Triton kernel implementations, or both.

Infrastructure: Triton (already pip-installable), GPU profiling tools (torch.profiler, built into PyTorch), and a kernel test harness that verifies correctness (numerical equivalence to the reference PyTorch implementation within fp16 tolerance) before performance evaluation. Validation: Measure both val_bpb and wall-clock training time. Compare three variants: (A) Python-only autoresearch, (B) kernel-only optimization (fix train.py, vary kernels), (C) co-optimization. The co-optimization variant should achieve better val_bpb per wall-clock-second. Also measure the kernel acceptance rate — what fraction of generated kernels are correct, and of those, what fraction are faster than PyTorch defaults?

### Third-Party Perspective

This is an ambitious and technically demanding idea. The potential impact is enormous — kernel-level optimization is responsible for some of the largest practical speedups in modern ML (FlashAttention, PagedAttention, fused optimizers). However, the implementation challenges are severe. Writing correct CUDA/Triton kernels is hard even for experts; having an LLM generate them introduces significant risk of subtle numerical bugs, race conditions, or memory safety issues. The correctness verification step is critical and must be extremely robust. Current LLMs can generate reasonable Triton kernels for simple operations but struggle with complex fusions. The approach also creates a tight coupling to specific GPU hardware, reducing portability. Despite these challenges, Triton's higher-level abstraction (compared to raw CUDA) makes this more feasible than it would have been a few years ago, and LLM kernel generation is an active research area with rapid improvements. This idea represents the logical end-state of autoresearch — optimizing the full stack from algorithm to hardware — but it's the hardest to implement well. High risk, very high reward.

---

## Idea 19: Novelty Search with Quality — Rewarding Behavioral Diversity, Not Just Performance

**TLDR:** Supplement the val_bpb objective with a novelty metric that rewards modifications producing models with behaviorally distinct outputs, preventing the optimization from collapsing to a narrow family of solutions and instead maintaining a diverse archive of qualitatively different approaches.

### The Idea

In evolutionary computation, a landmark finding was that explicitly rewarding *novelty* — how different an agent's behavior is from previously seen behaviors — often discovers better solutions than directly optimizing the objective function. This is because novelty search avoids the deceptive gradients and local optima that trap objective-driven search. The same principle applies to autoresearch: two train.py variants might have identical val_bpb but produce models with very different internal representations, output distributions, and generalization profiles. The current system has no way to value this diversity.

Novelty Search with Quality (NSQ) would maintain a behavioral archive: a collection of "behavioral signatures" from each trained model. A behavioral signature could be the model's output distribution on a fixed set of 1000 probe prompts — essentially a fingerprint of how the model behaves, independent of its aggregate val_bpb score. Each new modification is evaluated on two axes: (1) quality — did val_bpb improve? and (2) novelty — how different is this model's behavioral signature from all previously archived signatures? The acceptance criterion combines both: a modification with moderate val_bpb improvement but high novelty might be preferred over one with slightly better val_bpb but producing a behaviorally identical model.

This prevents the "convergent evolution" problem where autoresearch discovers one successful approach and then spends all subsequent iterations making minor variants of it. With novelty pressure, the system is incentivized to explore fundamentally different architectural and training strategies, even if they don't immediately beat the current best. This broader exploration has a higher chance of discovering radically better solutions that require traversing a behavioral valley — regions where val_bpb is temporarily worse but the qualitative behavior is stepping toward a better attractor basin.

The concept comes directly from Ken Stanley's work on novelty search and its successors (MAP-Elites, Quality-Diversity algorithms). These algorithms have been shown to outperform objective-only optimization in complex, deceptive fitness landscapes — which describes the space of possible training configurations.

### Implementation and Testing

Define a behavioral signature function: run the trained model on 1000 fixed probe sequences (diverse examples spanning code, math, dialogue, factual recall, creative writing). Record the top-5 token probabilities at each position, creating a ~50,000-dimensional vector. Compute pairwise novelty as the average L2 distance between the new model's signature and the k-nearest neighbors in the archive (k=15 is standard). Maintain an archive of up to 500 signatures, adding entries that are sufficiently novel (distance above a threshold). The combined fitness is: fitness = val_bpb_improvement + λ × novelty_score, where λ controls the exploration-exploitation tradeoff.

Infrastructure: A probe dataset (1000 fixed sequences, curated for behavioral diversity), a signature storage system (numpy arrays, ~200MB for 500 archived signatures), and a nearest-neighbor computation (fast with sklearn's BallTree for 50K dimensions). The probe evaluation adds ~30 seconds per iteration. Validation: Compare three variants over 300 iterations: (A) val_bpb only, (B) novelty only, (C) quality-diversity (combined). Measure final best val_bpb, archive coverage (how many distinct behavioral niches were explored), and robustness (variance of val_bpb across the top-10 archive members). The QD variant should produce a richer archive and discover solutions the pure val_bpb optimizer misses.

### Third-Party Perspective

Novelty search and quality-diversity algorithms are well-validated in evolutionary robotics and reinforcement learning, with strong theoretical and empirical support. Applying them to code-space optimization is natural and well-motivated. The main challenge is defining the "behavioral signature" — for language models, what constitutes meaningfully different behavior? Two models might produce different top-5 token distributions on probes but be functionally identical for downstream tasks, or vice versa. The signature design is critical and requires careful tuning. There's also computational cost: running 1000 probe sequences per iteration adds overhead, and storing/comparing 50K-dimensional signatures is non-trivial (though manageable with approximate nearest neighbors). The λ hyperparameter controlling quality-novelty tradeoff is another tuning challenge. Despite these concerns, the underlying insight is powerful: diversity is a resource, not a cost. Autoresearch that maintains diverse approaches has more options when the landscape changes and is less likely to get permanently stuck. Implementable with current technology and moderate engineering effort.

---

## Idea 20: Meta-Autoresearch — The Agent That Optimizes the Optimization Loop Itself

**TLDR:** Add an outer meta-loop that treats the entire autoresearch harness (prompt templates, acceptance criteria, evaluation protocol, iteration structure) as a modifiable program, optimizing the *research process* itself rather than just the training code — a system that learns how to learn better.

### The Idea

Autoresearch optimizes train.py but treats everything else as fixed: the LLM prompt that generates modifications, the acceptance criterion (val_bpb threshold), the evaluation protocol (how long to train, how to measure), the iteration structure (propose → train → evaluate → accept/reject), and the context provided to the agent. But these meta-parameters profoundly affect performance. A slightly different prompt might elicit better modifications. A different evaluation duration might give more reliable signal. A modified acceptance criterion (e.g., Idea 16's simulated annealing) might escape local optima. Why should these be fixed by a human designer when the system could optimize them too?

Meta-Autoresearch adds an outer optimization loop. The inner loop is standard autoresearch: modify train.py, train, evaluate, accept/reject. The outer loop modifies the *inner loop's configuration*: the system prompt template, the number of training steps per evaluation, the acceptance threshold, the temperature of the LLM's sampling, the amount of context (code history) provided, and the structure of the modification proposal. The outer loop evaluates its changes by running the inner loop for K iterations and measuring the *rate of improvement* — not the absolute val_bpb, but how quickly the inner loop is finding gains.

This is a genuinely recursive self-improvement system. The inner loop improves the model. The outer loop improves the inner loop. You could even add a third level that optimizes the outer loop's meta-parameters, though in practice two levels likely suffice before the signal becomes too noisy. The concept is inspired by meta-learning (learning to learn) and by the observation that in human research, the most impactful contributions often aren't new results but new *methods* — better ways of doing research that accelerate everyone's work.

The philosophical significance is profound: this is a system that can improve its own ability to improve. The fixed elements of autoresearch — designed by human engineers — represent a ceiling on what the system can discover. Meta-autoresearch removes that ceiling by making the research methodology itself subject to optimization.

### Implementation and Testing

Define a configuration file `harness_config.yaml` that parameterizes the autoresearch loop: LLM prompt template (with slots for code, history, instructions), sampling temperature, number of training steps, acceptance criterion type (greedy, annealing, threshold), context window allocation (how much history to include), and evaluation protocol (single run vs. averaged over N seeds). The outer loop proposes changes to this config, runs the inner loop for K=50 iterations, measures the improvement rate (total val_bpb delta / K), and accepts or rejects the meta-modification.

Infrastructure: The harness config system (a YAML file + config loader), and enough compute to run the inner loop multiple times per outer iteration (50 inner iterations × multiple outer iterations). The outer loop LLM can be the same model or a different one. Validation: Run meta-autoresearch for 10 outer iterations (500 total inner iterations) and compare the final val_bpb against standard autoresearch running for the same 500 iterations with a fixed harness. Also inspect the evolved harness configuration — did the system discover non-obvious improvements to the research process? Track the improvement rate over time: meta-autoresearch should show accelerating returns as the harness improves.

### Third-Party Perspective

This is the most conceptually ambitious idea in the collection — genuine recursive self-improvement. The concept is sound and draws from meta-learning, AutoML, and the theory of self-improving systems. The main challenges are severe: (1) the outer loop's signal is extremely noisy — "rate of improvement over 50 iterations" is a high-variance estimator, making it hard to distinguish good meta-changes from lucky runs; (2) the meta-search space is enormous (prompt templates alone have billions of possible variations) and the evaluation cost per meta-step is very high (50 full inner iterations); (3) there's a risk of the system finding degenerate configurations (e.g., an acceptance criterion that accepts everything, making "improvement rate" look high while actually degrading quality). Strong guardrails are needed: the outer loop should only modify within safe bounds, and each meta-change should be validated against a holdout benchmark.

Despite these challenges, even crude meta-optimization could yield significant gains. Human-designed research harnesses are almost certainly suboptimal — prompt wording, evaluation duration, and temperature settings are typically set by intuition, not data. This is the idea that most directly answers the question "what comes after autoresearch?" — the answer is autoresearch that improves autoresearch. Partially implementable today, though the full recursive vision requires careful engineering to prevent degenerate solutions and manage compute costs.

---

## Idea 21: Failure Pattern Mining — Learning More from Rejected Modifications Than Accepted Ones

**TLDR:** Systematically analyze the ~80-90% of modifications that autoresearch *rejects*, clustering them into failure modes and extracting negative transfer rules ("never do X when Y is true"), turning the vast graveyard of failed experiments into the system's most valuable learning resource.

### The Idea

Autoresearch discards failed modifications and moves on. But the rejected experiments contain enormously more information than the accepted ones — there are typically 5-10× more failures than successes, and each failure tells the system something about the structure of the optimization landscape. A modification that worsens val_bpb by 0.05 is just as informative as one that improves it by 0.05, but only the latter gets preserved. This is like a scientist throwing away all negative results — it's wasteful and leads to repeating the same mistakes.

Failure Pattern Mining would maintain a structured database of all rejected modifications, annotated with: the type of change, the magnitude of degradation, the state of train.py at the time, and any diagnostic information (training curves, gradient statistics if available from Idea 10). Periodically, the system would run a clustering analysis over the failure database to identify recurring failure patterns: "Increasing model width without proportionally increasing depth always hurts at this scale," "Swapping to cosine learning rate schedule fails when batch size is below 64," "Adding dropout to attention layers never helps for models under 50M parameters."

These mined rules become *negative constraints* in the agent's prompt: "Based on 23 prior experiments, do NOT propose modifications of type X under condition Y." This prevents the agent from repeatedly rediscovering the same dead ends. Over hundreds of iterations, the negative constraint set becomes a comprehensive map of "what doesn't work here" — which is often more useful than knowing what does work, because the space of bad ideas is vastly larger than the space of good ones.

The approach draws from inductive logic programming and negative mining in recommendation systems. In recommendation, knowing what a user *doesn't* like is often more informative than knowing what they do like, because dislikes constrain the space more efficiently. The same principle applies to code modifications.

### Implementation and Testing

Store every rejected modification in a `failures.jsonl` file with fields: iteration number, diff summary, modification category, val_bpb delta, train.py state hash, and any available diagnostics. Every 50 iterations, run a clustering analysis: group failures by modification category and current model state, identify clusters with >5 members (recurring failure patterns), and extract a natural language rule summarizing each cluster. Add the top-5 most robust negative rules to the agent's prompt as constraints.

Infrastructure: A `failures.jsonl` file (grows linearly, ~1KB per entry), a clustering module (k-means or DBSCAN on modification feature vectors), and an LLM-based rule summarizer that converts cluster statistics into natural language constraints. Validation: Compare modification acceptance rates between an agent with failure-mined constraints vs. a baseline agent over 200 iterations. The constrained agent should have a higher acceptance rate (fewer wasted iterations on known-bad modifications) and reach the same val_bpb with fewer total iterations. Also verify that the constraints don't over-restrict: measure whether any accepted modifications would have been blocked by the constraints (false positive rate should be <5%).

### Third-Party Perspective

This is an underrated, highly practical idea. The insight that rejected experiments are an information goldmine is correct and under-exploited in most optimization systems. The implementation is straightforward and low-overhead. The main risks are: (1) failure patterns are context-dependent — a modification that fails at iteration 50 might succeed at iteration 500 when the model has changed, so constraints based on early failures could become stale; (2) the clustering quality depends on how modifications are featurized, and LLM-generated code diffs are hard to cluster meaningfully; (3) over-aggressive constraints could prevent the agent from exploring modifications that are superficially similar to past failures but substantively different. Mitigation: expire constraints after N iterations and re-mine, and add a small probability of ignoring constraints (similar to epsilon-greedy exploration). The idea pairs beautifully with Idea 17 (Hypothesis Journal) — failed hypotheses become the raw material for negative rules. Fully implementable today with minimal infrastructure. The ratio of value to implementation effort is among the highest in this collection.

---

## Idea 22: Surrogate Model Prefiltering — Train a Cheap Predictor of Modification Success

**TLDR:** Train a lightweight neural network to predict whether a proposed code modification will improve val_bpb *before* actually running the expensive training loop, using the accumulated history of (modification, outcome) pairs as training data — filtering out likely failures at near-zero cost.

### The Idea

Each autoresearch iteration costs ~5 minutes of GPU time for training and evaluation, and ~80-90% of proposed modifications are rejected. That means 4-4.5 minutes of every 5 are wasted on modifications that don't help. If we could predict with even moderate accuracy which modifications will fail *before* running them, we could skip the losers and dramatically increase the system's effective iteration rate.

A Surrogate Model would be a small, cheap-to-evaluate neural network trained to predict val_bpb delta from a representation of the proposed code modification. The input would be an embedding of the code diff (computed by the LLM or a dedicated code embedding model) concatenated with features of the current train.py state (model size, optimizer type, current val_bpb, iteration number). The output would be a predicted val_bpb delta. The model is trained on the growing dataset of (modification_embedding, outcome) pairs from all prior autoresearch iterations.

The surrogate operates as a cheap prefilter. The LLM proposes a batch of K modifications (say K=10) instead of one. The surrogate scores all K, and only the top-scoring modification actually enters the expensive train-and-evaluate loop. This effectively gives the system 10× the proposal bandwidth at minimal additional cost (LLM inference for 10 proposals + 10 surrogate forward passes is trivially cheap compared to one 5-minute training run).

This is directly analogous to surrogate-assisted optimization in Bayesian optimization and evolutionary strategies, where expensive fitness evaluations are filtered through cheap approximations. The technique is standard in engineering optimization (e.g., aerodynamic design, drug discovery) where each evaluation costs hours of simulation. Autoresearch's 5-minute training runs are the same bottleneck at a different scale.

### Implementation and Testing

Collect a dataset from past autoresearch runs: for each iteration, store (code_diff_embedding, train_py_state_features, val_bpb_delta). The code diff embedding can be the mean-pooled last hidden state from a code embedding model (CodeBERT, StarEncoder) applied to the unified diff. Train a 2-layer MLP (input: 768-dim embedding + 20 state features, hidden: 256, output: 1 scalar) to predict val_bpb delta. Retrain the surrogate every 50 iterations as more data accumulates.

At each iteration, have the LLM generate K=10 candidate modifications. Embed each diff, score with the surrogate, and select the top-1 (or top-2 for diversity) for actual training. Infrastructure: A code embedding model (can run on CPU, inference is ~10ms per diff), a small MLP (trivial to train, <1 second), and a modification history database. Validation: After accumulating 100+ data points, measure the surrogate's ranking accuracy (does it correctly rank the best modification out of 10?) using leave-one-out cross-validation. Compare total val_bpb improvement per wall-clock hour between surrogate-filtered autoresearch and standard autoresearch. The filtered version should achieve the same improvement in significantly less time.

### Third-Party Perspective

Surrogate-assisted optimization is a mature field with decades of successful application. The transfer to code modification prediction is novel but well-motivated. The main challenge is whether code diffs are predictable enough — the relationship between a code change and its val_bpb impact is highly nonlinear and context-dependent. A small MLP might not capture these complex interactions, leading to poor predictions. However, even a surrogate that's only slightly better than random at filtering (say, 60% accuracy at identifying the best of 10 candidates vs. 10% for random) would provide substantial speedup. The surrogate also improves over time as it accumulates more training data, creating a virtuous cycle. The batch-proposal approach (generate 10, pick 1) is a clever way to get more value from cheap LLM inference. The main risk is the cold-start problem: the surrogate needs ~50-100 data points before it's useful, during which the system runs standard autoresearch. This is acceptable since the data accumulates naturally. Fully implementable today with off-the-shelf components. This is a strong "free lunch" idea — it adds minimal complexity and cost while potentially multiplying the effective iteration rate.

---

## Idea 23: Cross-Task Transfer — One Agent Optimizes Training Across Multiple Domains Simultaneously

**TLDR:** Run autoresearch on multiple tasks in parallel (language modeling, code generation, math reasoning, image captioning) and identify modifications that improve performance *across* tasks, discovering universal training principles rather than task-specific tricks.

### The Idea

Autoresearch currently optimizes training for a single task: language modeling measured by val_bpb on one dataset. Discoveries made in this narrow context may not generalize — a trick that helps predict English web text might be useless for code or mathematical reasoning. Conversely, the most valuable training insights are often *universal*: things like "warmup followed by cosine decay works well" or "RMSNorm is better than LayerNorm" hold across diverse tasks. But you can only discover universal principles by testing across multiple tasks.

Cross-Task Transfer autoresearch would maintain multiple parallel training pipelines — for example, English web text (val_bpb), Python code (code_bpb), mathematical reasoning (math_bpb), and multilingual text (multi_bpb). Each iteration, the agent proposes a modification and the system evaluates it on *all* tasks. Modifications are scored on a composite: average improvement across tasks, with a bonus for consistency (improving all tasks) and a penalty for task-specificity (improving one while hurting others). This naturally selects for modifications that discover universal training principles.

The approach is inspired by multi-task learning, where training on multiple objectives simultaneously often improves generalization on each individual objective. But here, multi-task optimization is applied at the *meta* level — it's not the model being trained on multiple tasks, but the *research process* being evaluated across multiple tasks. A modification that improves training for code AND math AND language is almost certainly discovering something fundamental about how neural networks learn, not exploiting a quirk of one dataset.

The side benefit is a ranked catalog of training techniques annotated with their generality: "Technique X: +0.03 bpb on language, +0.02 on code, +0.04 on math, 0.00 on multilingual → highly general" vs. "Technique Y: +0.05 on language, -0.01 on code, 0.00 on math, -0.02 on multilingual → English-specific." This catalog is directly useful for practitioners choosing which techniques to apply to their specific use case.

### Implementation and Testing

Set up 4 parallel training pipelines with different datasets: FineWeb (English), The Stack (code), OpenWebMath (math), and CulturaX (multilingual). Each pipeline uses the same base train.py with the same model architecture. Each iteration, the agent proposes one modification to the shared train.py. Run 5-minute training on all 4 datasets in parallel. Compute per-task val_bpb deltas and aggregate: score = mean(deltas) + λ × min(deltas), where the min term penalizes modifications that hurt any task.

Infrastructure: 4× GPU compute (but the training runs are embarrassingly parallel), 4 tokenized datasets, and a modified evaluation harness that collects multi-task results. Validation: After 200 iterations, compare: (1) the generality of discovered modifications (what fraction improve all 4 tasks?) between cross-task autoresearch and single-task autoresearch; (2) performance on a held-out 5th task (e.g., dialogue) — modifications discovered via cross-task optimization should transfer better to unseen tasks than those from single-task optimization, since they're selecting for universality.

### Third-Party Perspective

This is a well-motivated idea that addresses a genuine limitation of single-task optimization: it produces overfitted, non-generalizable discoveries. The multi-task evaluation framework is sound and draws from established multi-task learning principles. The main challenge is cost: 4× GPU compute per iteration is significant. Mitigation: the 4 runs are independent and fully parallelizable, so wall-clock time doesn't increase if you have 4 GPUs. The scoring function design is critical — the balance between average improvement and consistency must be tuned to avoid either accepting mediocre across-the-board modifications or rejecting brilliant task-specific ones. There's also a philosophical question: are universal training principles actually what we want? Sometimes the best approach is genuinely task-specific, and forcing universality could handicap optimization. The answer depends on the use case — if you're building a general-purpose foundation model, universal principles are exactly right; if you're building a specialized code model, they might not be. Despite this caveat, the universal-principle discovery angle is compelling and the technique catalog side product is independently valuable. Implementable today with sufficient compute.
