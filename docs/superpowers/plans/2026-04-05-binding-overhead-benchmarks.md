# Binding Overhead Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the performance cost of `mcts-dynamic` (trait-object, String-based) vs native Rust (monomorphized, zero-cost types) by benchmarking identical games through both paths, including a realistic Mancala game.

**Architecture:** A single benchmark file `mcts-dynamic/benches/overhead.rs` implements Mancala (Kalah rules, 4 stones/pit) in both native `GameState` and dynamic `GameCallbacks` forms, plus a trivial CountingGame baseline. Criterion groups produce side-by-side comparisons. The native side uses `AlphaGoPolicy` (same as dynamic) so the only difference is the binding layer.

**Tech Stack:** Rust, criterion 0.5, mcts (core), mcts-dynamic

---

## File Structure

| File | Responsibility |
|------|---------------|
| `mcts-dynamic/benches/overhead.rs` | All benchmark code: Mancala game (native + dynamic), CountingGame (native + dynamic), criterion groups |
| `mcts-dynamic/Cargo.toml` | Add criterion dev-dependency and `[[bench]]` section |

Everything lives in one file. The game implementations are benchmark-internal (not exported). This keeps the diff small and self-contained.

---

## Mancala (Kalah) Rules Reference

Standard Kalah with 6 pits per side, configurable initial stones (default 4):

```
       P2's pits (indices 7-12, right to left from P2's view)
  [12] [11] [10] [ 9] [ 8] [ 7]
[13]                            [ 6]
  [ 0] [ 1] [ 2] [ 3] [ 4] [ 5]
       P1's pits (indices 0-5, left to right)

Index 6  = P1's store
Index 13 = P2's store
```

- **Move:** Pick a non-empty pit on your side. Sow stones counterclockwise, skipping opponent's store.
- **Capture:** If last stone lands in an empty pit on your side AND the opposite pit has stones, capture both into your store.
- **Extra turn:** If last stone lands in your own store, you go again.
- **Terminal:** When one side is empty, the other player collects remaining stones into their store. Higher store wins.

Branching factor: 1-6 per turn. Average game length: ~40-60 moves. State: 14 `u8` values + current player.

---

### Task 1: Add criterion to mcts-dynamic

**Files:**
- Modify: `mcts-dynamic/Cargo.toml`

- [ ] **Step 1: Add criterion dev-dependency and bench config**

Add to `mcts-dynamic/Cargo.toml`:

```toml
[dev-dependencies]
serde_json = "1"
criterion = "0.5"

[[bench]]
name = "overhead"
harness = false
```

- [ ] **Step 2: Create empty benchmark file**

Create `mcts-dynamic/benches/overhead.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| b.iter(|| 1 + 1));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
```

- [ ] **Step 3: Verify it compiles and runs**

Run: `cargo bench -p mcts-dynamic --bench overhead -- --test`
Expected: Compiles, runs the placeholder benchmark, exits.

- [ ] **Step 4: Commit**

```bash
git add mcts-dynamic/Cargo.toml mcts-dynamic/benches/overhead.rs
git commit -m "scaffold: criterion benchmark for mcts-dynamic overhead tests"
```

---

### Task 2: Implement native Mancala (GameState)

**Files:**
- Modify: `mcts-dynamic/benches/overhead.rs`

- [ ] **Step 1: Write a basic test for Mancala move logic**

Add to `overhead.rs` (replacing placeholder content but keeping criterion imports):

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use mcts::tree_policy::AlphaGoPolicy;
use mcts::*;

// -----------------------------------------------------------------------
// Mancala (Kalah) — native implementation
// -----------------------------------------------------------------------

/// Kalah board: pits[0..6] = P1 side + store, pits[7..14] = P2 side + store.
/// pits[6] = P1 store, pits[13] = P2 store.
#[derive(Clone, PartialEq)]
struct Mancala {
    pits: [u8; 14],
    current: u8, // 0 = P1, 1 = P2
}

#[derive(Clone, Debug, PartialEq)]
struct MancalaMove(u8); // pit index (0-5 for P1, 7-12 for P2)

impl std::fmt::Display for MancalaMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum MancalaPlayer {
    P1,
    P2,
}

impl Mancala {
    fn new(stones_per_pit: u8) -> Self {
        let mut pits = [stones_per_pit; 14];
        pits[6] = 0;  // P1 store
        pits[13] = 0; // P2 store
        Self { pits, current: 0 }
    }

    fn my_store(&self) -> usize {
        if self.current == 0 { 6 } else { 13 }
    }

    fn opp_store(&self) -> usize {
        if self.current == 0 { 13 } else { 6 }
    }

    fn my_pits(&self) -> std::ops::Range<usize> {
        if self.current == 0 { 0..6 } else { 7..13 }
    }

    fn opp_pits(&self) -> std::ops::Range<usize> {
        if self.current == 0 { 7..13 } else { 0..6 }
    }

    fn side_empty(&self, range: std::ops::Range<usize>) -> bool {
        range.into_iter().all(|i| self.pits[i] == 0)
    }

    /// Sow stones from `pit`. Returns true if the player gets an extra turn.
    fn sow(&mut self, pit: usize) -> bool {
        let stones = self.pits[pit];
        self.pits[pit] = 0;
        let opp_store = self.opp_store();
        let my_store = self.my_store();
        let mut pos = pit;
        for _ in 0..stones {
            pos = (pos + 1) % 14;
            if pos == opp_store {
                pos = (pos + 1) % 14;
            }
            self.pits[pos] += 1;
        }
        // Capture: last stone in empty own pit, opposite has stones
        if pos != my_store {
            let my_range = self.my_pits();
            if my_range.contains(&pos) && self.pits[pos] == 1 {
                let opposite = 12 - pos;
                if self.pits[opposite] > 0 {
                    self.pits[my_store] += self.pits[opposite] + 1;
                    self.pits[pos] = 0;
                    self.pits[opposite] = 0;
                }
            }
        }
        // Extra turn if last stone landed in own store
        pos == my_store
    }

    fn collect_remaining(&mut self) {
        let p1_remaining: u8 = (0..6).map(|i| self.pits[i]).sum();
        let p2_remaining: u8 = (7..13).map(|i| self.pits[i]).sum();
        self.pits[6] += p1_remaining;
        self.pits[13] += p2_remaining;
        for i in 0..6 {
            self.pits[i] = 0;
        }
        for i in 7..13 {
            self.pits[i] = 0;
        }
    }
}

impl GameState for Mancala {
    type Move = MancalaMove;
    type Player = MancalaPlayer;
    type MoveList = Vec<MancalaMove>;

    fn current_player(&self) -> MancalaPlayer {
        if self.current == 0 {
            MancalaPlayer::P1
        } else {
            MancalaPlayer::P2
        }
    }

    fn available_moves(&self) -> Vec<MancalaMove> {
        if self.side_empty(self.my_pits()) || self.side_empty(self.opp_pits()) {
            return vec![];
        }
        self.my_pits()
            .filter(|&i| self.pits[i] > 0)
            .map(MancalaMove)
            .collect()
    }

    fn make_move(&mut self, mov: &MancalaMove) {
        let extra_turn = self.sow(mov.0 as usize);
        // Check if either side is now empty → game over
        if self.side_empty(0..6) || self.side_empty(7..13) {
            self.collect_remaining();
            return;
        }
        if !extra_turn {
            self.current = 1 - self.current;
        }
    }
}
```

- [ ] **Step 2: Add MCTS config and evaluator for native Mancala**

Append after the `GameState` impl:

```rust
struct MancalaEval;

impl Evaluator<NativeMancalaMCTS> for MancalaEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &Mancala,
        moves: &Vec<MancalaMove>,
        _: Option<SearchHandle<NativeMancalaMCTS>>,
    ) -> (Vec<f64>, i64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        // Heuristic: difference in stores
        let value = state.pits[6] as i64 - state.pits[13] as i64;
        (priors, value)
    }

    fn interpret_evaluation_for_player(
        &self,
        evaln: &i64,
        player: &MancalaPlayer,
    ) -> i64 {
        match player {
            MancalaPlayer::P1 => *evaln,
            MancalaPlayer::P2 => -*evaln,
        }
    }

    fn evaluate_existing_state(
        &self,
        _: &Mancala,
        evaln: &i64,
        _: SearchHandle<NativeMancalaMCTS>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct NativeMancalaMCTS;

impl MCTS for NativeMancalaMCTS {
    type State = Mancala;
    type Eval = MancalaEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::Ignore
    }
    fn fpu_value(&self) -> f64 {
        0.0
    }
}
```

- [ ] **Step 3: Add a quick sanity-test benchmark to verify the game works**

Replace the placeholder benchmark function with:

```rust
fn bench_mancala_native(c: &mut Criterion) {
    c.bench_function("mancala native 10k", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                Mancala::new(4),
                NativeMancalaMCTS,
                MancalaEval,
                AlphaGoPolicy::new(1.5),
                (),
            );
            mcts.playout_n(10_000);
        });
    });
}

criterion_group!(benches, bench_mancala_native);
criterion_main!(benches);
```

- [ ] **Step 4: Verify it compiles and runs**

Run: `cargo bench -p mcts-dynamic --bench overhead -- --test`
Expected: Compiles, runs the benchmark, exits.

- [ ] **Step 5: Commit**

```bash
git add mcts-dynamic/benches/overhead.rs
git commit -m "bench: native Mancala (Kalah) game implementation for overhead comparison"
```

---

### Task 3: Implement dynamic Mancala (GameCallbacks)

**Files:**
- Modify: `mcts-dynamic/benches/overhead.rs`

- [ ] **Step 1: Add the dynamic Mancala implementation**

Add after the native Mancala code, before the benchmark functions:

```rust
use mcts_dynamic::*;

// -----------------------------------------------------------------------
// Mancala (Kalah) — dynamic implementation (same logic, GameCallbacks API)
// -----------------------------------------------------------------------

#[derive(Clone)]
struct DynMancala {
    pits: [u8; 14],
    current: u8,
}

impl DynMancala {
    fn new(stones_per_pit: u8) -> Self {
        let mut pits = [stones_per_pit; 14];
        pits[6] = 0;
        pits[13] = 0;
        Self { pits, current: 0 }
    }

    fn my_store(&self) -> usize {
        if self.current == 0 { 6 } else { 13 }
    }

    fn opp_store(&self) -> usize {
        if self.current == 0 { 13 } else { 6 }
    }

    fn my_pits(&self) -> std::ops::Range<usize> {
        if self.current == 0 { 0..6 } else { 7..13 }
    }

    fn side_empty(&self, range: std::ops::Range<usize>) -> bool {
        range.into_iter().all(|i| self.pits[i] == 0)
    }

    fn sow(&mut self, pit: usize) -> bool {
        let stones = self.pits[pit];
        self.pits[pit] = 0;
        let opp_store = self.opp_store();
        let my_store = self.my_store();
        let mut pos = pit;
        for _ in 0..stones {
            pos = (pos + 1) % 14;
            if pos == opp_store {
                pos = (pos + 1) % 14;
            }
            self.pits[pos] += 1;
        }
        if pos != my_store {
            let my_range = self.my_pits();
            if my_range.contains(&pos) && self.pits[pos] == 1 {
                let opposite = 12 - pos;
                if self.pits[opposite] > 0 {
                    self.pits[my_store] += self.pits[opposite] + 1;
                    self.pits[pos] = 0;
                    self.pits[opposite] = 0;
                }
            }
        }
        pos == my_store
    }

    fn collect_remaining(&mut self) {
        let p1_remaining: u8 = (0..6).map(|i| self.pits[i]).sum();
        let p2_remaining: u8 = (7..13).map(|i| self.pits[i]).sum();
        self.pits[6] += p1_remaining;
        self.pits[13] += p2_remaining;
        for i in 0..6 { self.pits[i] = 0; }
        for i in 7..13 { self.pits[i] = 0; }
    }
}

impl GameCallbacks for DynMancala {
    fn clone_box(&self) -> Box<dyn GameCallbacks> {
        Box::new(self.clone())
    }

    fn current_player(&self) -> i32 {
        self.current as i32
    }

    fn available_moves(&self) -> Vec<String> {
        if self.side_empty(self.my_pits())
            || self.side_empty(if self.current == 0 { 7..13 } else { 0..6 })
        {
            return vec![];
        }
        self.my_pits()
            .filter(|&i| self.pits[i] > 0)
            .map(|i| i.to_string())
            .collect()
    }

    fn make_move(&mut self, mov: &str) {
        let pit: usize = mov.parse().unwrap();
        let extra_turn = self.sow(pit);
        if self.side_empty(0..6) || self.side_empty(7..13) {
            self.collect_remaining();
            return;
        }
        if !extra_turn {
            self.current = 1 - self.current;
        }
    }
}

struct DynMancalaEval;

impl EvalCallbacks for DynMancalaEval {
    fn evaluate(&self, state: &dyn GameCallbacks, moves: &[String]) -> (Vec<f64>, f64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        // We can't access pits directly through the trait, so return 0.
        // This is fine — both sides use the same eval quality; we're measuring overhead.
        (priors, 0.0)
    }
}
```

**Important note:** The dynamic evaluator returns `0.0` value instead of the store-difference heuristic because `EvalCallbacks::evaluate` only gets a `&dyn GameCallbacks` — it can't downcast to access `pits`. This is actually realistic: in a real binding, the evaluator would call back into the host language. For a fair comparison, we should also set the native evaluator to return `0.0`. Update `MancalaEval::evaluate_new_state` to return `(priors, 0)` instead of `(priors, value)`.

- [ ] **Step 2: Update native MancalaEval to match**

Change the native evaluator to also return `0` as value (so both sides have identical eval quality):

```rust
// In MancalaEval::evaluate_new_state, change the return to:
        (priors, 0)
```

- [ ] **Step 3: Add the dynamic Mancala benchmark**

Add after `bench_mancala_native`:

```rust
fn bench_mancala_dynamic(c: &mut Criterion) {
    c.bench_function("mancala dynamic 10k", |b| {
        b.iter(|| {
            let mut mgr = DynMCTSManager::new(
                Box::new(DynMancala::new(4)),
                Box::new(DynMancalaEval),
                DynConfig {
                    exploration_constant: 1.5,
                    fpu_value: 0.0,
                    ..DynConfig::default()
                },
            );
            mgr.playout_n(10_000);
        });
    });
}
```

Update the criterion group:

```rust
criterion_group!(benches, bench_mancala_native, bench_mancala_dynamic);
```

- [ ] **Step 4: Verify both benchmarks compile and run**

Run: `cargo bench -p mcts-dynamic --bench overhead -- --test`
Expected: Both benchmarks run.

- [ ] **Step 5: Commit**

```bash
git add mcts-dynamic/benches/overhead.rs
git commit -m "bench: dynamic Mancala for native-vs-dynamic overhead comparison"
```

---

### Task 4: Add CountingGame baseline benchmarks

**Files:**
- Modify: `mcts-dynamic/benches/overhead.rs`

- [ ] **Step 1: Add native CountingGame PUCT benchmark**

Add before the criterion group:

```rust
// -----------------------------------------------------------------------
// CountingGame — trivial baseline (isolates pure binding overhead)
// -----------------------------------------------------------------------

#[derive(Clone)]
struct CountingGame(i64);

#[derive(Clone, Debug, PartialEq)]
enum CountingMove {
    Add,
    Sub,
}

impl std::fmt::Display for CountingMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CountingMove::Add => write!(f, "Add"),
            CountingMove::Sub => write!(f, "Sub"),
        }
    }
}

impl GameState for CountingGame {
    type Move = CountingMove;
    type Player = ();
    type MoveList = Vec<CountingMove>;

    fn current_player(&self) -> Self::Player {}
    fn available_moves(&self) -> Vec<CountingMove> {
        if self.0 == 100 {
            vec![]
        } else {
            vec![CountingMove::Add, CountingMove::Sub]
        }
    }
    fn make_move(&mut self, mov: &CountingMove) {
        match mov {
            CountingMove::Add => self.0 += 1,
            CountingMove::Sub => self.0 -= 1,
        }
    }
}

struct CountingEval;

impl Evaluator<NativeCountingMCTS> for CountingEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        _state: &CountingGame,
        moves: &Vec<CountingMove>,
        _: Option<SearchHandle<NativeCountingMCTS>>,
    ) -> (Vec<f64>, i64) {
        let n = moves.len();
        let priors = if n > 0 { vec![1.0 / n as f64; n] } else { vec![] };
        (priors, 0)
    }
    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }
    fn evaluate_existing_state(
        &self,
        _: &CountingGame,
        evaln: &i64,
        _: SearchHandle<NativeCountingMCTS>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct NativeCountingMCTS;

impl MCTS for NativeCountingMCTS {
    type State = CountingGame;
    type Eval = CountingEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::Ignore
    }
    fn fpu_value(&self) -> f64 {
        0.0
    }
}

fn bench_counting_native(c: &mut Criterion) {
    c.bench_function("counting native 100k", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                CountingGame(0),
                NativeCountingMCTS,
                CountingEval,
                AlphaGoPolicy::new(1.5),
                (),
            );
            mcts.playout_n(100_000);
        });
    });
}
```

- [ ] **Step 2: Add dynamic CountingGame benchmark**

```rust
#[derive(Clone)]
struct DynCountingGame(i64);

impl GameCallbacks for DynCountingGame {
    fn clone_box(&self) -> Box<dyn GameCallbacks> {
        Box::new(self.clone())
    }
    fn current_player(&self) -> i32 {
        0
    }
    fn available_moves(&self) -> Vec<String> {
        if self.0 == 100 {
            vec![]
        } else {
            vec!["Add".to_string(), "Sub".to_string()]
        }
    }
    fn make_move(&mut self, mov: &str) {
        match mov {
            "Add" => self.0 += 1,
            "Sub" => self.0 -= 1,
            _ => panic!("unknown move"),
        }
    }
}

struct DynCountingEval;

impl EvalCallbacks for DynCountingEval {
    fn evaluate(&self, _state: &dyn GameCallbacks, moves: &[String]) -> (Vec<f64>, f64) {
        let n = moves.len();
        let priors = if n > 0 { vec![1.0 / n as f64; n] } else { vec![] };
        (priors, 0.0)
    }
}

fn bench_counting_dynamic(c: &mut Criterion) {
    c.bench_function("counting dynamic 100k", |b| {
        b.iter(|| {
            let mut mgr = DynMCTSManager::new(
                Box::new(DynCountingGame(0)),
                Box::new(DynCountingEval),
                DynConfig {
                    exploration_constant: 1.5,
                    fpu_value: 0.0,
                    ..DynConfig::default()
                },
            );
            mgr.playout_n(100_000);
        });
    });
}
```

- [ ] **Step 3: Update criterion group**

```rust
criterion_group!(
    benches,
    bench_counting_native,
    bench_counting_dynamic,
    bench_mancala_native,
    bench_mancala_dynamic,
);
criterion_main!(benches);
```

- [ ] **Step 4: Verify all four benchmarks run**

Run: `cargo bench -p mcts-dynamic --bench overhead -- --test`
Expected: 4 benchmarks compile and run.

- [ ] **Step 5: Commit**

```bash
git add mcts-dynamic/benches/overhead.rs
git commit -m "bench: CountingGame baseline for native-vs-dynamic overhead comparison"
```

---

### Task 5: Add parallel Mancala benchmarks

**Files:**
- Modify: `mcts-dynamic/benches/overhead.rs`

- [ ] **Step 1: Add parallel benchmarks**

Add before the criterion group:

```rust
fn bench_mancala_native_parallel(c: &mut Criterion) {
    c.bench_function("mancala native 10k 4-thread", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                Mancala::new(4),
                NativeMancalaMCTS,
                MancalaEval,
                AlphaGoPolicy::new(1.5),
                (),
            );
            mcts.playout_n_parallel(10_000, 4);
        });
    });
}

fn bench_mancala_dynamic_parallel(c: &mut Criterion) {
    c.bench_function("mancala dynamic 10k 4-thread", |b| {
        b.iter(|| {
            let mut mgr = DynMCTSManager::new(
                Box::new(DynMancala::new(4)),
                Box::new(DynMancalaEval),
                DynConfig {
                    exploration_constant: 1.5,
                    fpu_value: 0.0,
                    ..DynConfig::default()
                },
            );
            mgr.playout_n_parallel(10_000, 4);
        });
    });
}
```

- [ ] **Step 2: Update criterion group**

```rust
criterion_group!(
    benches,
    bench_counting_native,
    bench_counting_dynamic,
    bench_mancala_native,
    bench_mancala_dynamic,
    bench_mancala_native_parallel,
    bench_mancala_dynamic_parallel,
);
criterion_main!(benches);
```

- [ ] **Step 3: Verify all six benchmarks run**

Run: `cargo bench -p mcts-dynamic --bench overhead -- --test`
Expected: 6 benchmarks compile and run.

- [ ] **Step 4: Run the actual benchmarks and record results**

Run: `cargo bench -p mcts-dynamic --bench overhead`

Expected output format (example — actual numbers will vary):
```
counting native 100k    time: [XXX ms YYY ms ZZZ ms]
counting dynamic 100k   time: [XXX ms YYY ms ZZZ ms]
mancala native 10k      time: [XXX ms YYY ms ZZZ ms]
mancala dynamic 10k     time: [XXX ms YYY ms ZZZ ms]
mancala native 10k 4t   time: [XXX ms YYY ms ZZZ ms]
mancala dynamic 10k 4t  time: [XXX ms YYY ms ZZZ ms]
```

Compute overhead ratios:
- `counting dynamic / counting native` = worst-case overhead (trivial game)
- `mancala dynamic / mancala native` = realistic overhead
- `mancala dynamic 4t / mancala native 4t` = parallel overhead

- [ ] **Step 5: Commit**

```bash
git add mcts-dynamic/benches/overhead.rs
git commit -m "bench: parallel Mancala + complete 6-benchmark overhead suite"
```

- [ ] **Step 6: Run `cargo clippy -p mcts-dynamic --all-targets` and fix any warnings**

Run: `cargo clippy -p mcts-dynamic --all-targets`
Expected: 0 warnings

---

## Verification

After all tasks:

```bash
cargo bench -p mcts-dynamic --bench overhead -- --test   # all 6 benchmarks compile
cargo test -p mcts-dynamic                                # existing 26 tests still pass
cargo clippy -p mcts-dynamic --all-targets                # 0 warnings
```

To run the full benchmark suite and see overhead ratios:

```bash
cargo bench -p mcts-dynamic --bench overhead
```

## Interpreting Results

| Ratio (dynamic / native) | Interpretation |
|--------------------------|----------------|
| < 1.5x | Negligible overhead. Bindings are viable even for simple games. |
| 1.5x - 3x | Moderate. Acceptable for most real games. |
| 3x - 10x | Significant. Fine for NN-guided search (eval dominates), poor for rollout-heavy. |
| > 10x | Severe. Investigate: String allocation, Box cloning, or REWARD_SCALE issues. |

The **Mancala ratio** is the number that matters for real users. The **CountingGame ratio** is the theoretical ceiling that shrinks as game complexity increases.
