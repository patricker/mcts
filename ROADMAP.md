# MCTS — Roadmap

Five foundational features. All complete.

---

## 1. Seeded RNG

**Status:** Done

`PolicyRng` uses `SmallRng` (seedable, `Send`, faster than `ThreadRng`).
Optional seed via `MCTS::rng_seed()` — each thread gets a deterministic
RNG from `base_seed + thread_id`.

### Implementation

| File | What |
|------|------|
| `src/tree_policy.rs` | `PolicyRng` wraps `SmallRng` with `seeded(u64)` constructor |
| `src/lib.rs` | `fn rng_seed(&self) -> Option<u64>` on MCTS trait |
| `src/search_tree.rs` | `make_thread_data()` seeds per-thread via `thread_counter` |
| `tests/mcts_tests.rs` | `SeededMCTS`, determinism test (same seed → same visit counts) |

---

## 2. Dirichlet Noise + FPU + Temperature Selection

**Status:** Done

Three independent additions that together unlock the AlphaZero search paradigm.

### 2a. Dirichlet Root Noise

`MCTS::dirichlet_noise() -> Option<(f64, f64)>` returns `(epsilon, alpha)`.
Applied in `SearchTree::new()` and `advance_root()` via
`TreePolicy::apply_dirichlet_noise()`. Gamma sampler (Marsaglia-Tsang +
Ahrens-Dieter boost) implemented in `tree_policy.rs` — no external deps.

### 2b. First Play Urgency (FPU)

`MCTS::fpu_value() -> f64` (default `INFINITY` = try all children first).
Both `UCTPolicy` and `AlphaGoPolicy` use it for zero-visit children.

### 2c. Temperature-Based Move Selection

`MCTS::selection_temperature() -> f64` (default `0.0` = argmax).
`MCTSManager::best_move()` samples proportional to `visits^(1/tau)` via
`RefCell<SmallRng>`. `principal_variation()` always uses argmax.

### Implementation

| File | What |
|------|------|
| `src/lib.rs` | `dirichlet_noise()`, `fpu_value()`, `selection_temperature()` on MCTS trait |
| `src/tree_policy.rs` | `apply_dirichlet_noise()`, `sample_gamma()`, `sample_dirichlet()` |
| `src/search_tree.rs` | Noise applied in `new()` and `advance_root()` |
| `tests/mcts_tests.rs` | FPU (3), Dirichlet (5), temperature (4) tests |

### References

- [AlphaGo Zero (Silver et al., 2017)](https://www.nature.com/articles/nature24270)
- [AlphaZero (Silver et al., 2018)](https://www.science.org/doi/10.1126/science.aar6404)
- [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)

---

## 3. Batched Neural Network Evaluation

**Status:** Done

### Architecture: Bridge Pattern

`BatchedEvaluatorBridge<Spec, B>` implements `Evaluator` by routing
`evaluate_new_state` calls through an `mpsc` channel to a dedicated
collector thread. Search threads block on `sync_channel(1)` oneshot
responses. **Zero changes to `SearchTree`, `MoveInfo`, or the playout loop.**

```
Search threads ──► evaluate_new_state() ──► [mpsc send + block]
                                                    │
                                              [Collector Thread]
                                              gather up to max_batch_size
                                              call evaluate_batch()
                                              distribute results
                                                    │
Search threads ◄── [sync_channel recv] ◄────────────┘
```

Users set `type Eval = BatchedEvaluatorBridge<MyMCTS, MyBatchEval>` and
use existing `playout_n_parallel()` etc. — the bridge is transparent.

### Implementation

| File | What |
|------|------|
| `src/batch.rs` (new, 251 lines) | `BatchEvaluator` trait, `BatchConfig`, `BatchedEvaluatorBridge`, `collector_loop`, `Drop` impl |
| `src/lib.rs` | `pub mod batch; pub use batch::*;` (2 lines) |
| `tests/mcts_tests.rs` | Mock batch evaluators (UCT + AlphaGo), 10 tests |

**Zero new dependencies.** Uses `std::sync::mpsc`, `Mutex`, `Arc`.

### References

- [KataGo NNEvaluator](https://github.com/lightvector/KataGo/blob/master/cpp/search/search.cpp)
- [Leela Chess Zero](https://github.com/LeelaChessZero/lc0)
- [Batch MCTS (Cazenave, 2020)](https://www.lamsade.dauphine.fr/~cazenave/papers/BatchMCTS.pdf)

---

## 4. MCTS-Solver

**Status:** Done

### Algorithm

Proven game-theoretic values (win/loss/draw) propagate up the tree during
backpropagation. Solved subtrees are skipped during selection.

- `ProvenValue` enum (`Unknown/Win/Loss/Draw`) stored as `AtomicU8` on `SearchNode`
- Convention: value is from the `current_player()`'s perspective at each node
- Opt-in via `MCTS::solver_enabled()` (default `false`, zero overhead when off)
- Terminal classification via `GameState::terminal_value()`

### Propagation Rules (minimax)

- ANY child `Loss` (opponent loses) → parent is `Win`
- ALL children `Win` (opponent wins everywhere) → parent is `Loss`
- All proven, mix of Win+Draw → parent is `Draw`
- Any unexpanded child → parent stays `Unknown`

### Selection Policy

When solver enabled, both `UCTPolicy` and `AlphaGoPolicy` score:
- Child `Loss` → `f64::INFINITY` (parent wins by choosing this)
- Child `Win` → `f64::NEG_INFINITY` (avoid — opponent wins)
- `select_child_after_search` prefers proven-win children
- `playout()` returns `false` when root is proven (stop searching)

### Implementation

| File | What |
|------|------|
| `src/lib.rs` | `ProvenValue` enum, `GameState::terminal_value()`, `MCTS::solver_enabled()`, solver-aware `select_child_after_search`, `root_proven_value()` |
| `src/search_tree.rs` | `proven: AtomicU8` on `SearchNode`, `child_proven_value()` on `MoveInfo`, `try_prove_node()`, `propagate_proven()` in `finish_playout`, early exit in `playout()` |
| `src/tree_policy.rs` | Conditional proven-value scoring in both policies |
| `tests/mcts_tests.rs` | TinyNim (two-player Nim) game, 13 tests: forced wins/losses, exhaustive positions 1-6, proven root stops search, parallel correctness |

### References

- [MCTS-Solver (Winands et al., 2008)](https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf)
- [Score-Bounded MCTS (Cazenave & Saffidine)](https://www.lamsade.dauphine.fr/~cazenave/papers/mcsolver.pdf)
- [OpenSpiel MCTS solver](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/algorithms/mcts.h)

---

## 5. Chance Nodes

**Status:** Done

### Architecture: Open-Loop MCTS

Chance outcomes are sampled during playouts but NOT stored as separate
tree nodes. The tree only contains decision nodes. Different playouts
through the same move edge sample different outcomes — node statistics
converge to the expected value over the distribution. **Zero changes to
`SearchNode`, `MoveInfo`, tree policies, or backpropagation.**

### Design

`GameState::chance_outcomes()` returns `Option<Vec<(Self::Move, f64)>>`.
Outcomes reuse the `Move` type — games encode chance events as move
variants (e.g., `Die(u8)`). Applied via `make_move()`.

In the playout loop, after each `state.make_move()`:
```rust
while let Some(outcomes) = state.chance_outcomes() {
    let outcome = sample_chance_outcome(&outcomes, chance_rng);
    state.make_move(outcome);
}
```

Also resolved at playout start (handles root in pending-chance state).

Separate `chance_rng: SmallRng` on `ThreadDataFull` — avoids coupling
with tree policy RNG, seeded deterministically when `rng_seed()` is set.

### Implementation

| File | What |
|------|------|
| `src/lib.rs` | `GameState::chance_outcomes()` default method, `chance_rng` on `ThreadDataFull` |
| `src/search_tree.rs` | `sample_chance_outcome()` helper, chance resolution in `playout()` (after make_move + at start), seed `chance_rng` in `make_thread_data()` |
| `tests/mcts_tests.rs` | DiceGame (single-player d3), 8 tests: optimal strategy, expected values, seeded determinism, parallel, backward compat |

### References

- [Chance nodes in MCTS (Schadd et al., 2012)](https://dke.maastrichtuniversity.nl/m.winands/documents/schadd_2012_stochastic.pdf)
- [OpenSpiel chance nodes](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h)
- [Stochastic MuZero (Antonoglou et al., 2022)](https://openreview.net/forum?id=X6D9bAHhBQ1)

---

## Summary

All 5 items complete. 80 tests, 0 warnings, 0 new runtime dependencies.

```
1. Seeded RNG ✓
   └── 2. Dirichlet + FPU + Temperature ✓
        └── 3. Batched NN Eval ✓

4. MCTS-Solver ✓
5. Chance Nodes ✓
```
