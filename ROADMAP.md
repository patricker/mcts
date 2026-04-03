# MCTS — Roadmap

Five foundational features, ordered by dependency. Each one unlocks multiple
downstream capabilities.

---

## 1. Seeded RNG

**Status:** Done
**Effort:** Small (1-2 hours)
**Unlocks:** Deterministic testing of every future feature

### Problem

`PolicyRng` in `src/tree_policy.rs` wraps `rand::thread_rng()` (ThreadRng),
which is non-deterministic. Two runs of the same search produce different
visit distributions. This makes it impossible to write exact-value assertions
or reproduce bugs.

### Design

Add an optional seed to the MCTS configuration. When set, each thread gets a
deterministic RNG seeded from `base_seed + thread_id`.

```rust
// MCTS trait addition
fn rng_seed(&self) -> Option<u64> { None }

// PolicyRng changes
pub struct PolicyRng {
    rng: SmallRng,  // was ThreadRng -- SmallRng is seedable + Send
}

impl PolicyRng {
    pub fn from_seed(seed: u64) -> Self {
        Self { rng: SmallRng::seed_from_u64(seed) }
    }
    pub fn new() -> Self {
        Self { rng: SmallRng::from_rng(&mut rand::rng()) }
    }
}
```

`SmallRng` is `Send` (ThreadRng is not), faster, and seedable. The switch
from ThreadRng to SmallRng is beneficial regardless of seeding.

### Changes

| File | Change |
|------|--------|
| `src/tree_policy.rs` | Replace `ThreadRng` with `SmallRng`, add `from_seed` constructor |
| `src/lib.rs` | Add `fn rng_seed(&self) -> Option<u64>` to MCTS trait |
| `src/search_tree.rs` | Pass seed to thread-local PolicyRng creation in `playout()` |
| `tests/mcts_tests.rs` | Add determinism test: same seed → same visit counts |

### References

- [`rand::rngs::SmallRng`](https://docs.rs/rand/latest/rand/rngs/struct.SmallRng.html) — fast, seedable, `Send`
- [Reproducibility in AlphaZero](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — KataGo seeds its RNG per-thread

---

## 2. Dirichlet Noise + FPU + Temperature Selection

**Status:** Not started
**Effort:** Small-Medium (3-5 hours for all three)
**Unlocks:** AlphaZero self-play, Gumbel-Top-k, pondering, any NN-guided training loop

These are three small, independent additions that together unlock the entire
neural-network-guided search paradigm. Every AlphaZero/MuZero implementation
requires all three.

### 2a. Dirichlet Root Noise

Mix the neural prior with Dirichlet noise at the root to ensure exploration
of all moves during self-play:

```
noisy_prior[i] = (1 - eps) * prior[i] + eps * Dir(alpha)[i]
```

Typical values: `eps = 0.25`, `alpha = 0.03` (Go), `alpha = 0.3` (Chess).

**Design:** Add a method to the MCTS trait and apply noise in `create_node`
when the node is the root.

```rust
// MCTS trait addition
fn dirichlet_noise(&self) -> Option<(f64, f64)> { None }  // (epsilon, alpha)
```

The `create_node` function already has access to the move evaluations. When
`dirichlet_noise()` returns `Some`, sample from `Dir(alpha)` and blend.

### 2b. First Play Urgency (FPU)

Configurable default value for unvisited children. Currently UCT returns
`f64::INFINITY` for unvisited nodes, forcing all children to be tried before
any is revisited. With neural priors, this wastes simulations on low-prior
moves.

```rust
// MCTS trait addition
fn fpu_value(&self) -> f64 { f64::INFINITY }  // default: current behavior
// AlphaZero typical: parent_value - fpu_reduction (e.g., 0.0 or -0.2)
```

**Changes:** `UCTPolicy::choose_child` and `AlphaGoPolicy::choose_child`
use `fpu_value()` instead of `f64::INFINITY` for zero-visit children.

### 2c. Temperature-Based Move Selection

After search, select moves proportional to `visits^(1/tau)` rather than
argmax by visits. tau=1 for exploration (training), tau→0 for exploitation
(play).

```rust
// MCTS trait addition
fn selection_temperature(&self) -> f64 { 0.0 }  // 0.0 = argmax (current behavior)

// Alternative select_child_after_search when temperature > 0:
// weight[i] = visits[i]^(1/tau)
// select proportional to weight[i] / sum(weights)
```

**Design:** Override `select_child_after_search` default to check
`selection_temperature()`. When 0, use current argmax. When > 0, sample
proportionally. Requires seeded RNG (item 1).

### References

- [AlphaGo Zero (Silver et al., 2017)](https://www.nature.com/articles/nature24270) — Dirichlet noise, temperature selection
- [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404) — FPU, noise, temperature
- [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) — Production implementation of all three
- [Leela Chess Zero](https://lczero.org/dev/backend/nn/) — FPU reduction tuning
- [OpenSpiel MCTS](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/mcts.h) — `dirichlet_alpha`, `uct_c` (FPU via exploration constant)

---

## 3. Batched Neural Network Evaluation

**Status:** Not started
**Effort:** Large (8-12 hours)
**Unlocks:** AlphaZero, MuZero, LLM reasoning at scale, GPU utilization

### Problem

The current `Evaluator` trait evaluates one leaf at a time, synchronously.
Neural network inference is 10-100x faster when batched. Without batching,
each search thread blocks on a single forward pass — the GPU is idle 90% of
the time.

### Architecture

The standard approach (KataGo, Leela Chess Zero, LightZero):

1. Search threads hit a leaf and enqueue it (state + callback channel)
2. A dedicated evaluator thread collects N leaves (or waits for a timeout)
3. The evaluator runs one batched forward pass
4. Results are distributed back to waiting threads via channels

```
Thread 1: select → expand → [enqueue leaf] → wait → backprop
Thread 2: select → expand → [enqueue leaf] → wait → backprop
Thread 3: select → expand → [enqueue leaf] → wait → backprop
                              ↓
                     [Batch Evaluator Thread]
                     collect N leaves
                     model.forward(batch)
                     distribute results
```

### Design

Add a `BatchEvaluator` trait alongside the existing `Evaluator`:

```rust
pub trait BatchEvaluator<Spec: MCTS>: Send + Sync {
    type StateEvaluation: Sync + Send;

    fn evaluate_batch(
        &self,
        states: &[Spec::State],
    ) -> Vec<(Vec<MoveEvaluation<Spec>>, Self::StateEvaluation)>;

    fn interpret_evaluation_for_player(
        &self,
        evaluation: &Self::StateEvaluation,
        player: &Player<Spec>,
    ) -> i64;
}
```

The playout loop changes from synchronous evaluation to a
suspend/enqueue/resume pattern. Implementation options:

**Option A: Channel-based.** Each thread sends its leaf state through an
`mpsc` channel to the batch evaluator, then blocks on a `oneshot` channel
for the result. Simple, uses standard library primitives.

**Option B: Virtual loss + retry.** Thread marks the leaf with virtual loss,
enqueues it, and immediately starts a new playout (without waiting). When the
batch result arrives, it's written to the node asynchronously. Next time a
thread visits that node, it finds the evaluation. More complex but higher
throughput — no thread ever blocks.

**Recommendation:** Start with Option A (channel-based). It's simpler, correct,
and sufficient for most use cases. Option B is an optimization for later.

### Changes

| File | Change |
|------|--------|
| `src/lib.rs` | Add `BatchEvaluator` trait, add `MCTS::BatchEval` associated type |
| `src/search_tree.rs` | Add batch playout mode with channel-based leaf collection |
| `src/lib.rs` | Add `MCTSManager::playout_batched(batch_size, num_threads)` |
| New: `src/batch.rs` | Batch collection queue, evaluator thread loop |
| `tests/mcts_tests.rs` | Test with mock batch evaluator |

### References

- [KataGo NNEvaluator](https://github.com/lightvector/KataGo/blob/master/cpp/search/search.cpp) — Production batch evaluation with virtual loss
- [Leela Chess Zero batching](https://github.com/LeelaChessZero/lc0) — Batch collection + NN backend abstraction
- [LightZero](https://github.com/opendilab/LightZero) — Python batch MCTS with multiple Zero variants
- [Batch MCTS (Cazenave, 2020)](https://www.lamsade.dauphine.fr/~cazenave/papers/BatchMCTS.pdf) — Formal treatment of batched evaluation
- [Adaptive Parallelism for DNN-MCTS (2023)](https://arxiv.org/html/2310.05313) — 13x acceleration via batching
- [MuZero (Schrittwieser et al., 2020)](https://www.nature.com/articles/s41586-020-03051-4) — Learned model + batched search

---

## 4. MCTS-Solver

**Status:** Not started
**Effort:** Medium (4-6 hours)
**Unlocks:** Score-Bounded MCTS, PN-MCTS hybrids, endgame solvers, smarter time management

### Problem

When a terminal state is reached during a playout, the library records the
evaluation and backpropagates it as a regular score. But a terminal win is
*proven* — no amount of additional search will change it. The library
continues allocating simulations to subtrees that are already decided.

### Algorithm

During backpropagation, when a terminal state is reached, propagate
game-theoretic proven values up the tree:

1. A terminal node is proven (win/loss/draw for each player)
2. A node is a **proven loss** if the opponent has ANY proven-win child
3. A node is a **proven win** if ALL opponent children are proven losses
4. A node is a **proven draw** if all children are proven draws/losses and at least one is a draw

Solved nodes are skipped during selection — the tree policy treats them as
having infinite/negative-infinite value as appropriate.

```rust
#[derive(Clone, Copy, PartialEq)]
pub enum ProvenValue {
    Unknown,
    Win,
    Loss,
    Draw,
}

// Added to SearchNode (or NodeStats):
proven: AtomicU8,  // maps to ProvenValue
```

### Generalization: Score-Bounded MCTS

Instead of just win/loss/draw, track upper and lower bounds on the evaluation
for arbitrary scoring games. A node's upper bound is the max of its
children's upper bounds; its lower bound is the max of its children's lower
bounds (from the current player's perspective). When upper_bound ==
lower_bound, the node is solved.

[[Cazenave & Saffidine]](https://www.lamsade.dauphine.fr/~cazenave/papers/mcsolver.pdf)

### Changes

| File | Change |
|------|--------|
| `src/search_tree.rs` | Add `proven: AtomicU8` to `SearchNode`, update `finish_playout` to propagate proven values |
| `src/search_tree.rs` | Modify selection to skip proven-loss children, prefer proven-win children |
| `src/lib.rs` | Add `fn is_terminal(&self) -> Option<TerminalValue>` to `GameState` (or infer from empty moves + evaluation) |
| `src/tree_policy.rs` | UCT/PUCT handle proven nodes (infinite UCB for proven wins, skip proven losses) |
| `tests/mcts_tests.rs` | Test: proven win propagates, proven subtrees not re-searched, CountingGame(99) solved after one Add playout |

### References

- [MCTS-Solver (Winands, Bjornsson, Saito, 2008)](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf) — Original paper, 96% win rate vs vanilla UCT in Lines of Action
- [Score-Bounded MCTS (Cazenave & Saffidine)](https://www.lamsade.dauphine.fr/~cazenave/papers/mcsolver.pdf) — Generalization to scoring games
- [PN-MCTS (2023)](https://arxiv.org/abs/2303.09449) — Hybrid with Proof Number Search
- [Generalized PN-MCTS (2025)](https://arxiv.org/html/2506.13249) — Latest extension
- [OpenSpiel MCTS solver](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/mcts.h) — `solve` flag enables proven value propagation

---

## 5. Chance Nodes

**Status:** Not started
**Effort:** Medium (4-6 hours)
**Unlocks:** Stochastic games, ISMCTS, robotics, autonomous driving, Open-Loop MCTS, belief-state MCTS

### Problem

The library assumes deterministic transitions: `make_move()` always produces
the same next state. This blocks an entire class of domains:

- **Stochastic games:** Backgammon (dice), Poker (card draws), Catan (resource rolls)
- **Robotics:** Sensor noise, actuator uncertainty
- **Planning under uncertainty:** Weather, traffic, market dynamics

### Algorithm

Add a new node type — **chance nodes** — where the transition is sampled
from a distribution rather than chosen by a player. The tree alternates
between decision nodes (player chooses) and chance nodes (nature samples):

```
Decision Node (player picks action)
  └── Chance Node (nature samples outcome)
        ├── Outcome A (p=0.3) → Decision Node
        ├── Outcome B (p=0.5) → Decision Node
        └── Outcome C (p=0.2) → Decision Node
```

At chance nodes, the tree policy doesn't select — it samples from the
outcome distribution. Backpropagation weights updates by outcome probability.

### Design

Extend `GameState` with an optional stochastic transition:

```rust
pub trait GameState: Clone {
    type Move: Sync + Send + Clone;
    type Player: Sync;
    type MoveList: IntoIterator<Item = Self::Move>;
    type Outcome: Sync + Send + Clone;

    fn current_player(&self) -> Self::Player;
    fn available_moves(&self) -> Self::MoveList;
    fn make_move(&mut self, mov: &Self::Move);

    // --- New: stochastic transitions ---
    /// If the current state requires a chance event (dice roll, card draw),
    /// return the possible outcomes with their probabilities.
    /// Default: None (deterministic transition).
    fn chance_outcomes(&self) -> Option<Vec<(Self::Outcome, f64)>> { None }

    /// Apply a chance outcome to the state.
    fn apply_outcome(&mut self, outcome: &Self::Outcome) {
        let _ = outcome;  // default: no-op (deterministic games)
    }
}
```

For backward compatibility, `type Outcome = ()` and `chance_outcomes()`
returns `None` by default. Existing games are unaffected.

In the playout loop, after `make_move()`, check `chance_outcomes()`. If
`Some`, sample an outcome proportional to probabilities and call
`apply_outcome()`. In the tree, chance nodes store outcomes as children
(keyed by outcome value, weighted by probability).

### Changes

| File | Change |
|------|--------|
| `src/lib.rs` | Add `Outcome` type, `chance_outcomes()`, `apply_outcome()` to `GameState` |
| `src/search_tree.rs` | Add `ChanceNode` variant or flag on `SearchNode`, handle in `playout()` and `descend()` |
| `src/search_tree.rs` | Chance-node backpropagation: weight by outcome probability |
| `tests/mcts_tests.rs` | Test with simple dice game (e.g., roll 1-6, score = sum) |

### References

- [Chance nodes in MCTS (Schadd et al., 2012)](https://dke.maastrichtuniversity.nl/m.winands/documents/schadd_2012_stochastic.pdf) — Original treatment
- [OpenSpiel chance nodes](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h) — `ChanceNode` as a player type
- [Stochastic MuZero (Antonoglou et al., 2022)](https://openreview.net/forum?id=X6D9bAHhBQ1) — Learned chance outcomes
- [MBAPPE (autonomous driving)](https://arxiv.org/abs/2309.08452) — MCTS with stochastic prediction models
- [MCTS.jl DPW](https://juliapomdp.github.io/MCTS.jl/latest/dpw/) — Double progressive widening for stochastic transitions
- [ISMCTS (Cowling et al., 2012)](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) — Requires chance/determinization support

---

## Dependency Graph

```
1. Seeded RNG
   └── 2. Dirichlet + FPU + Temperature (needs seeded RNG for temperature sampling)
        └── 3. Batched NN Eval (needs FPU/noise for AlphaZero pattern)

4. MCTS-Solver (independent)
   └── Score-Bounded MCTS, PN-MCTS hybrids, endgame solvers

5. Chance Nodes (independent)
   └── ISMCTS, belief-state MCTS, stochastic games, robotics
```

Items 1→2→3 form a chain. Items 4 and 5 are independent and can be built in
parallel with 1-3.

**Suggested execution order:** 1, then 2 + 4 in parallel, then 3 + 5 in
parallel.
