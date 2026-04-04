# MCTS — Vision Document

*What someone imagines when they find this repo.*

---

> **What this document is.** You've just found this repo. You read the README,
> ran the example, glanced at the benchmarks. This document captures what runs
> through your head in the next ten minutes — the research you'd implement, the
> systems you'd build, the reason you'd choose this library over writing your
> own.
>
> It is not a roadmap. It is not a promise. It is the library the field has
> been waiting for.

---

## The Thirty-Second Pitch

You see the trait:

```rust
impl GameState for YourDomain {
    fn current_player(&self) -> Player { /* who moves next */ }
    fn available_moves(&self) -> Vec<Action> { /* legal actions */ }
    fn make_move(&mut self, action: &Action) { /* apply transition */ }
}
```

Three methods. Your domain plugs in. The library handles lock-free parallel
search, neural network batching, and tree policies from UCT to Gumbel-Top-k.

Then the punchline: **10M nodes/second on a single core, and it compiles to
WASM.** Bindings for Python, C, C++, JavaScript, Java, and C#. One
implementation. Every language. Every domain.

Standard PUCT *degrades* with more compute budget in some domains. Gumbel
selection restores monotonic scaling
([[ReSCALE, ICAPS 2026]](https://arxiv.org/abs/2603.21162)). This library
ships both — and lets you swap policies in one line.

|  | This library | OpenSpiel | LightZero | MCTS.jl |
|---|:---:|:---:|:---:|:---:|
| Lock-free parallel | Yes | No | No | No |
| NN batch eval | Yes | Partial | Yes | No |
| WASM target | Yes | No | No | No |
| Tree re-rooting | Yes | No | No | No |
| Language bindings | 7 | 2 | 1 | 1 |

You star the repo. And you stop writing your own.

---

## 1. LLM Reasoning & Code Generation

> *"MCTS is becoming the System 2 reasoning engine for large language models."*

This is the largest and fastest-growing MCTS use case. The library's trait
design maps naturally onto it: `available_moves()` generates candidate
next-steps via LLM sampling, `make_move()` appends to the reasoning trace,
and the `Evaluator` scores partial solutions.

```rust
impl GameState for ReasoningTrace {
    type Move = LLMCompletion;           // next reasoning step
    fn available_moves(&self) -> Vec<LLMCompletion> {
        sample_n_completions(&self.prompt, 5)  // expand 5 candidates
    }
    fn make_move(&mut self, step: &LLMCompletion) {
        self.steps.push(step.clone());   // extend the trace
    }
}
```

- **Process reward-guided selection.** Per-step value estimates guide which
  reasoning branches to explore, not just scoring complete solutions.
  [[ReST-MCTS*, NeurIPS 2024]](https://arxiv.org/abs/2406.07394)

- **Refinement trees.** Children aren't new actions — they're *refined
  versions* of the parent's answer. Selection -> Self-Refine -> Self-Evaluate
  -> Backpropagate. Achieved GPT-4-level math olympiad performance with
  LLaMA-3 8B.
  [[MCTSr, 2024]](https://arxiv.org/abs/2406.07394)

- **Dynamic action generation.** Actions created at expansion time by an LLM
  ("LLM-as-Action-Model"), not enumerated upfront. Progressive widening
  supports this natively — `max_children(visits)` controls how many
  LLM-generated candidates get search budget.
  [[Alpha-SQL, ICML 2025]](https://openreview.net/forum?id=kGg1ndttmI)

- **Adaptive branching.** At each node, dynamically decide whether to "go
  wider" (expand new children) or "go deeper" (refine existing ones).
  Extended to multi-LLM collective intelligence.
  [[AB-MCTS, NeurIPS 2025 Spotlight]](https://arxiv.org/abs/2503.04412)

- **Code generation & software engineering.** MCTS-guided generate/improve/fix
  cycles. SWE-Search achieves 23% improvement over standard agents on
  SWE-bench.
  [[SWE-Search, ICLR 2025]](https://openreview.net/forum?id=G7sIFXugTX),
  [[GIF-MCTS, NeurIPS 2024]](https://openreview.net/forum?id=9SpWvX9ykp)

- **Agentic workflows.** MCTS over operator sequences (Ensemble, Review,
  Revise, Code Execute) for automated agent design. Outperforms manual
  designs by 5.7%.
  [[AFlow, ICLR 2025 Oral]](https://openreview.net/forum?id=z5uVAKwmjf)

- **Theorem proving.** MCTS with proof-assistant feedback for formal
  mathematics. `available_moves()` returns tactic applications; the proof
  assistant provides ground-truth verification.
  [[DeepSeek-Prover-V1.5]](https://arxiv.org/abs/2408.15795),
  [[CARTS, ICLR 2025]](https://openreview.net/forum?id=VQwI055flA)

---

## 2. Neural Network-Guided Search (AlphaZero / MuZero)

> *"I have a policy+value network. I need MCTS that can batch leaf
> evaluations to my GPU."*

Neural network inference is 10-100x faster when batched. Without batching
infrastructure, NN-guided search is impractical. This is the single biggest
gap in most MCTS libraries.

```rust
impl BatchEvaluator<MyMCTS> for NNEvaluator {
    type StateEvaluation = Value;
    fn evaluate_batch(&self, leaves: &[(State, MoveList)]) -> Vec<(Vec<Policy>, Value)> {
        self.model.forward_batch(leaves)  // one GPU call for N leaves
    }
    // ...
}

// Wrap it and use as a normal Evaluator — the bridge handles batching transparently
let eval = BatchedEvaluatorBridge::new(NNEvaluator::new(), BatchConfig::default());
```

- **Batched async evaluation.** Multiple search threads collect leaf states
  into a queue. A separate evaluator thread processes them in batches and
  distributes results back. You provide the network; the library handles the
  plumbing.
  [[Schrittwieser et al., 2020]](https://www.nature.com/articles/s41586-020-03051-4)

- **Dirichlet root noise.** `(1-eps)*prior + eps*Dir(alpha)` at the root for
  self-play exploration. Table stakes for any AlphaZero implementation.

- **First Play Urgency (FPU).** Configurable default value for unvisited
  children. With neural priors, the prior should guide exploration — not a
  forced round-robin of every child.

- **Temperature-based move selection.** After search, select moves
  proportional to `visits^(1/tau)`. tau=1 for exploration, tau->0 for
  exploitation. Required for self-play training loops.

- **Gumbel-Top-k + Sequential Halving.** Sample K candidates, halve the worst
  at each stage. Matches AlphaZero quality with far fewer simulations.
  [[Danihelka et al., ICLR 2022]](https://openreview.net/forum?id=bERaNdoegnO)

*See also:* [LightZero](https://github.com/opendilab/LightZero) (Python, 8+ Zero variants),
[KataGo](https://github.com/lightvector/KataGo) (strongest open-source Go engine)

---

## 3. Board Games & General Game Playing

> *"I just want to build a strong game AI without reimplementing MCTS from
> scratch for the tenth time."*

The classic use case, handled end-to-end:

- **UCT, PUCT, RAVE/GRAVE.** The three policy families that cover 90% of
  game AI. Pick one, plug in your evaluator, search.

- **MCTS-Solver.** When a subtree is provably won or lost, stop wasting
  simulations on it. Proven values propagate via minimax. Win rates reach
  96% vs vanilla MCTS in Lines of Action at 10K simulations.
  [[Winands et al., 2008]](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf)

- **Transposition tables.** States reached via different move orders share
  search results. Ships with an approximate quadratic-probing hash table.

- **Pondering.** Search during the opponent's turn. When they move, re-root
  the tree via `advance_root` and keep searching. No work wasted.
  [[Baier & Winands, 2012]](https://dke.maastrichtuniversity.nl/m.winands/documents/time_management_for_monte_carlo_tree_search.pdf)

- **Time management.** Adaptive search budgets: spend less when the best move
  is clearly ahead, more when top moves are close. Early termination when
  continued search can't change the decision.

*See also:* [OpenSpiel](https://github.com/google-deepmind/open_spiel) (C++/Python, 90+ games),
[KataGo](https://github.com/lightvector/KataGo),
[Ludii](https://ludii.games/)

---

## 4. Combinatorial Optimization & Scheduling

> *"Scheduling, routing, molecular design — NP-hard problems where MCTS
> balances exploration and exploitation in massive discrete search spaces."*

MCTS isn't just for games. Any sequential decision problem with a large
search space and an evaluable intermediate state is a candidate.

- **Job shop scheduling.** Dynamic flexible job shop scheduling (DFJSP),
  hybrid flow shop, car manufacturing workshop scheduling.
  [[DyRo-MCTS, 2025]](https://arxiv.org/html/2509.21902)

- **Traveling salesman & graph problems.** Heatmap-guided MCTS for large-scale
  TSP, graph coloring, bin packing.
  [[NeurIPS 2024 Workshop]](https://openreview.net/forum?id=TMHOHRR0FA)

- **Drug discovery & molecular design.** De novo molecule generation with
  multi-objective Pareto search. `available_moves()` = next atom/bond;
  `Evaluator` = binding affinity + drug-likeness + synthesizability.
  [[ParetoDrug, 2024]](https://www.nature.com/articles/s42003-024-06746-w),
  [[Mothra, 2024]](https://pubs.acs.org/doi/10.1021/acs.jcim.4c00759)

- **Retrosynthetic planning.** Finding synthesis pathways from available
  building blocks — a natural tree search problem.
  [[EG-MCTS, 2023]](https://www.nature.com/articles/s42004-023-00911-8)

- **Chip design & PCB routing.** Distributed MCTS for placement and routing
  in electronic design automation.

- **Compiler & kernel optimization.** MCTS guides compiler transformation
  selection (tiling, fusion, vectorization) as a sequential decision process.
  [[OptiML, 2026]](https://arxiv.org/html/2602.12305)

- **Nested Rollout Policy Adaptation (NRPA).** Single-player MCTS variant
  that adapts rollout policies via gradient ascent. Holds world records in
  Morpion Solitaire and crossword puzzles.
  [[Cazenave, IJCAI 2011]](https://www.ijcai.org/Proceedings/11/Papers/115.pdf)

---

## 5. Robotics, Autonomous Driving & Continuous Domains

> *"My action space is continuous. My state transitions are stochastic. Most
> MCTS libraries can't handle either."*

- **Double progressive widening.** Extend progressive widening to both action
  AND state spaces. Actions sampled from continuous distributions rather than
  enumerated.
  [[Couetoux et al., 2011]](https://link.springer.com/chapter/10.1007/978-3-642-25566-3_40)

- **Chance nodes.** Stochastic transitions (dice rolls, card draws, sensor
  noise) modeled as chance nodes in the tree.

- **Autonomous driving.** MCTS plans lane changes, intersection navigation,
  and cooperative maneuvers without inter-vehicle communication.
  [[MBAPPE, 2023]](https://arxiv.org/abs/2309.08452)

- **Constraint-aware pruning.** Train a safety critic offline, prune unsafe
  trajectories during online search.
  [[C-MCTS, NeurIPS 2024 Workshop]](https://openreview.net/forum?id=BMw4Cm0gGO)

- **Open-Loop MCTS.** Eliminates state tracking from tree nodes for
  non-deterministic environments. Nodes represent action sequences, not
  states — significantly smaller trees with different accuracy tradeoffs.

- **Multi-objective search.** Track multiple objectives per node. Maintain a
  Pareto front during search. Hypervolume-based UCT for selection.

*See also:* [MCTS.jl](https://github.com/JuliaPOMDP/MCTS.jl) (Julia, DPW + POMDP integration),
[resibots/mcts](https://github.com/resibots/mcts) (C++, continuous domains)

---

## 6. Narrative, Creative & Generative AI

> *"The GM searches over possible story interventions and picks the one that
> maximizes dramatic tension."*

A Game Master for interactive fiction uses MCTS to search over narrative
interventions — character actions, plot twists, environmental changes — and
selects the one that best serves the story's dramatic needs.

- **Shallow search + rich evaluation.** Deeper MCTS can make decisions *worse*
  in some domains — narrative domains drift from plausibility with depth.
  `max_playout_depth` searches wide and shallow.
  [[Brockman & Saffidine, ICAPS 2024]](https://icaps24.icaps-conference.org/)

- **Progressive widening for huge action spaces.** Every character x every
  action x every target = combinatorial explosion. Evaluator-ranked move
  ordering focuses budget on the most promising interventions.

- **Tree re-rooting across ticks.** Pick an intervention, advance the
  simulation, continue searching from the preserved subtree.

- **Procedural content generation.** MCTS generates game levels (Sokoban
  puzzles, platformer levels), balancing playability constraints with variety.

- **Music composition.** Monophonic music generation with coherence scoring.
  Emotion-controlled symbolic music generation using PUCT search.
  [[IEEE, 2022]](https://ieeexplore.ieee.org/document/9751419/)

*See also:* [Narrative Studio](https://arxiv.org/html/2504.02426v1) (LLM + MCTS for branching narrative)

---

## 7. Imperfect Information & Multi-Agent

> *"Poker, Hanabi, strategy games with fog of war — half of interesting games
> have hidden information."*

- **Information Set MCTS (ISMCTS).** Build the tree over information sets
  (what the player knows), not exact game states. Each iteration samples a
  determinization consistent with observations, then runs standard MCTS.
  [[Cowling et al., 2012]](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf)

- **Decoupled UCT.** For simultaneous-move games, each player maintains
  separate UCB statistics. Joint actions formed by combining independent
  selections.

- **Belief-state MCTS.** For POMDPs, maintain a belief distribution over
  hidden state and search over belief-action pairs.

- **Dialogue planning.** Goal-oriented conversation control via MCTS.
  `available_moves()` = dialogue acts; `Evaluator` = probability of reaching
  the conversation goal.
  [[ChatSOP, ACL 2025]](https://aclanthology.org/2025.acl-long.863.pdf)

*See also:* [OpenSpiel](https://github.com/google-deepmind/open_spiel) (extensive imperfect-info support)

---

## 8. Architecture & Performance

> *"UCT was published in 2006. I have 64 cores. It's 2026. What's better?"*

**Parallelism.** The library ships lock-free tree parallelism with virtual
loss. But there's more:

- **Root parallelization (ensemble).** N independent trees, merge vote
  counts. 14.9x strength-speedup for 16 threads.
  [[Chaslot et al., 2008]](https://dke.maastrichtuniversity.nl/m.winands/documents/multithreadedMCTS2.pdf)

- **Speculative inter-decision parallelism.** Search future moves
  speculatively. 5.8x latency reduction in 9x9 Go training.
  [[Speculative MCTS, NeurIPS 2024]](https://openreview.net/forum?id=g1HxCIc0wi)

**Tree policies.** The `TreePolicy` trait means new policies are drop-in:

- **Thompson Sampling.** Sample from posterior distributions. No exploration
  constant to tune.
  [[Bai et al., NeurIPS 2013]](https://proceedings.neurips.cc/paper/2013/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html)
- **Boltzmann / DENTS.** Temperature-controlled softmax selection.
  [[NeurIPS 2023]](https://openreview.net/forum?id=NG4DaApavi)
- **Wasserstein MCTS.** Distributional backup via Wasserstein barycenter.
  [[Dam et al., ICML 2025]](https://openreview.net/forum?id=DUGFTH9W8B)
- **Score-Bounded MCTS.** Generalized MCTS-Solver tracking upper/lower bounds
  for arbitrary scoring games.
  [[Cazenave & Saffidine]](https://www.lamsade.dauphine.fr/~cazenave/papers/mcsolver.pdf)

**Memory & performance:**

- **Arena-based layout.** Contiguous node storage for 2.2-2.8x cache speedup.
  [[Array-Based MCTS, 2025]](https://arxiv.org/html/2508.20140v1)
- **Memory-based limits.** `max_memory_mb` alongside `node_limit`.
- **Search tree serialization.** Save/restore for distributed computation.
- **Structured diagnostics.** Nodes/sec, branching factor, depth stats.

---

## 9. Every Language, Every Platform

> *"The core is Rust. It compiles to everything."*

**Monorepo structure:**
```
mcts/
  core/              # Rust core library
  ffi/               # C FFI layer (cbindgen-generated header)
  clients/
    python/          # PyO3 + maturin -> PyPI
    js/              # wasm-bindgen + wasm-pack -> npm
    java/            # JNI bindings -> Maven
    csharp/          # P/Invoke -> NuGet
    c/               # Direct FFI consumption
    cpp/             # C++ wrapper header
  website/           # Docusaurus v3 documentation site
```

| Language | Binding Tech | Package Manager |
|----------|-------------|-----------------|
| Rust | Native | crates.io |
| Python | PyO3 + maturin | PyPI |
| JavaScript/TS | wasm-bindgen | npm (WASM) |
| C | cbindgen (cdylib) | Header + shared lib |
| C++ | C FFI + wrapper header | Header + shared lib |
| Java | JNI | Maven Central |
| C# | P/Invoke | NuGet |

**WASM for the doc site.** The Rust core compiles to WebAssembly. Interactive
examples run in the browser at native speed. Visualize the search tree
growing in real time. No server required.

---

## 10. World-Class Documentation

> *"The docs don't just describe the API. They teach you MCTS."*

- **Interactive tutorials.** Step through a playout visually. Watch selection,
  expansion, simulation, and backpropagation happen node by node. Powered by
  WASM — runs in the browser, no install.

- **Live playground.** Pick a game preset (Tic-Tac-Toe, Connect Four, 2048).
  Run MCTS. See the tree grow. Adjust exploration constants and watch the
  visit distribution change instantly.

- **Algorithm gallery.** Side-by-side comparison of UCT vs PUCT vs Thompson
  Sampling vs Gumbel. Same game, same budget. Animated tree visualizations.

- **Research citations.** Every feature links to the paper that introduced it:
  [UCT (Kocsis & Szepesvari, 2006)](https://link.springer.com/chapter/10.1007/11871842_29),
  [PUCT (Silver et al., 2016)](https://www.nature.com/articles/nature16961),
  [RAVE (Gelly & Silver, 2011)](https://www.cs.utexas.edu/~pstone/Courses/394Rspring13/resources/mcrave.pdf),
  [ISMCTS (Cowling et al., 2012)](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf),
  [Gumbel MuZero (Danihelka et al., 2022)](https://openreview.net/forum?id=bERaNdoegnO),
  [MCTS-Solver (Winands et al., 2008)](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf)

- **Cookbook.** Complete working code for:
  Connect Four AI, AlphaZero training loop, LLM reasoning with MCTS,
  narrative GM, card game AI with ISMCTS, scheduling optimization

- **Multi-language API reference.** Every public type with examples in Rust,
  Python, JavaScript, C++, Java, and C#. Generated from source, verified in CI.

---

## What Already Ships

What makes these visions feel *real* — not hand-wavy — is that the hard
parts are already done:

| Foundation | Status |
|---|---|
| Lock-free parallel tree search (virtual loss) | Ships |
| UCT + PUCT (AlphaGo) tree policies | Ships |
| Transposition table (approximate hash) | Ships |
| Tree re-rooting (`advance_root`) | Ships |
| Progressive widening (`max_children`) | Ships |
| Depth-limited search (`max_playout_depth`) | Ships |
| Statistics export (`root_child_stats`) | Ships |
| Async background search | Ships |
| Time-limited + count-limited search | Ships |
| Cycle detection + configurable behavior | Ships |
| Custom node data + evaluator traits | Ships |
| Full test coverage + criterion benchmarks | Ships |
| Seeded RNG (deterministic search) | Ships |
| Dirichlet root noise | Ships |
| First Play Urgency (FPU) | Ships |
| Temperature-based move selection | Ships |
| Batched neural network evaluation | Ships |
| MCTS-Solver (proven value propagation) | Ships |
| Chance nodes (open-loop stochastic transitions) | Ships |

Every vision above builds on something that already works. The architecture
(trait-based `GameState` + `Evaluator` + `TreePolicy` separation) is
designed for exactly this extensibility. New tree policies are drop-in.
New evaluation strategies are drop-in. New parallelism modes compose with
existing infrastructure.

---

*This document captures what Monte Carlo Tree Search looks like when someone
finally does it right. The field has shifted from game-tree search to
universal planning infrastructure — for LLM reasoning, molecular design,
autonomous driving, narrative AI, and software engineering. The library that
covers all of these, with Rust performance and bindings for every language,
doesn't exist yet. We're building it.*
