//! Binding overhead benchmarks: native (monomorphized) vs dynamic (trait-object) Mancala.
//!
//! Both implementations use identical Mancala (Kalah) rules with 6 pits and 4 stones.
//! The only differences are the type-level representation (enum vs String moves,
//! concrete vs trait-object state) to isolate the overhead of the dynamic adapter.

use criterion::{criterion_group, criterion_main, Criterion};
use mcts::tree_policy::AlphaGoPolicy;
use mcts::*;
use mcts_dynamic::*;

// ===========================================================================
// Mancala board layout (Kalah, 6 pits, 4 stones per pit)
//
//        P2's pits (indices 7-12)
//   [12] [11] [10] [ 9] [ 8] [ 7]
// [13]                            [ 6]
//   [ 0] [ 1] [ 2] [ 3] [ 4] [ 5]
//        P1's pits (indices 0-5)
//
// Index  6 = P1's store
// Index 13 = P2's store
// ===========================================================================

const PITS: usize = 6;
const STONES: u8 = 4;
const P1_STORE: usize = 6;
const P2_STORE: usize = 13;

/// Sow stones from `pit` for the given player (0=P1, 1=P2).
/// Returns the index where the last stone landed.
fn sow(pits: &mut [u8; 14], pit: usize, current: u8) -> usize {
    let stones = pits[pit];
    pits[pit] = 0;
    let skip_store = if current == 0 { P2_STORE } else { P1_STORE };
    let mut pos = pit;
    for _ in 0..stones {
        pos = (pos + 1) % 14;
        if pos == skip_store {
            pos = (pos + 1) % 14;
        }
        pits[pos] += 1;
    }
    pos
}

/// Check if a capture occurs and apply it. A capture happens when the last
/// stone lands in an empty pit on the current player's side, and the
/// opposite pit has stones.
fn maybe_capture(pits: &mut [u8; 14], last_pos: usize, current: u8) {
    let (own_range, store) = if current == 0 {
        (0..PITS, P1_STORE)
    } else {
        (7..7 + PITS, P2_STORE)
    };
    // The pit must have exactly 1 stone (the one just placed) and be on our side.
    if own_range.contains(&last_pos) && pits[last_pos] == 1 {
        let opposite = 12 - last_pos;
        if pits[opposite] > 0 {
            pits[store] += pits[opposite] + 1;
            pits[opposite] = 0;
            pits[last_pos] = 0;
        }
    }
}

/// Collect remaining stones into their respective stores when one side is empty.
/// Returns true if the game ended (one side was empty).
fn collect_remaining(pits: &mut [u8; 14]) -> bool {
    let p1_empty = pits[0..PITS].iter().all(|&s| s == 0);
    let p2_empty = pits[7..7 + PITS].iter().all(|&s| s == 0);
    if p1_empty {
        for i in 7..7 + PITS {
            pits[P2_STORE] += pits[i];
            pits[i] = 0;
        }
        true
    } else if p2_empty {
        for i in 0..PITS {
            pits[P1_STORE] += pits[i];
            pits[i] = 0;
        }
        true
    } else {
        false
    }
}

// ===========================================================================
// Task 2: Native Mancala (GameState)
// ===========================================================================

#[derive(Clone)]
struct Mancala {
    pits: [u8; 14],
    current: u8, // 0 = P1, 1 = P2
}

impl Mancala {
    fn new() -> Self {
        let mut pits = [STONES; 14];
        pits[P1_STORE] = 0;
        pits[P2_STORE] = 0;
        Self { pits, current: 0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct MancalaMove(u8);

#[derive(Clone, Debug, PartialEq)]
enum MancalaPlayer {
    P1,
    P2,
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
        let range = if self.current == 0 {
            0..PITS
        } else {
            7..7 + PITS
        };
        range
            .filter(|&i| self.pits[i] > 0)
            .map(|i| MancalaMove(i as u8))
            .collect()
    }

    fn make_move(&mut self, mov: &MancalaMove) {
        let pit = mov.0 as usize;
        let last_pos = sow(&mut self.pits, pit, self.current);

        // Extra turn: if last stone lands in own store, don't switch player
        let own_store = if self.current == 0 {
            P1_STORE
        } else {
            P2_STORE
        };
        if last_pos == own_store {
            // Keep current player, but still check if game ended
            collect_remaining(&mut self.pits);
            return;
        }

        maybe_capture(&mut self.pits, last_pos, self.current);
        collect_remaining(&mut self.pits);

        // Switch player
        self.current = 1 - self.current;
    }
}

// --- Native evaluator: uniform priors, value 0 ---

struct MancalaEval;

impl Evaluator<NativeMancalaMCTS> for MancalaEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        _state: &Mancala,
        moves: &Vec<MancalaMove>,
        _handle: Option<SearchHandle<NativeMancalaMCTS>>,
    ) -> (Vec<f64>, i64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        (priors, 0)
    }

    fn evaluate_existing_state(
        &self,
        _state: &Mancala,
        evaln: &i64,
        _handle: SearchHandle<NativeMancalaMCTS>,
    ) -> i64 {
        *evaln
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &MancalaPlayer) -> i64 {
        *evaln
    }
}

// --- Native MCTS config ---

#[derive(Default)]
struct NativeMancalaMCTS;

impl MCTS for NativeMancalaMCTS {
    type State = Mancala;
    type Eval = MancalaEval;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type TranspositionTable = ();
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::Ignore
    }

    fn fpu_value(&self) -> f64 {
        0.0
    }
}

// ===========================================================================
// Task 3: Dynamic Mancala (GameCallbacks)
// ===========================================================================

#[derive(Clone)]
struct DynMancala {
    pits: [u8; 14],
    current: u8,
}

impl DynMancala {
    fn new() -> Self {
        let mut pits = [STONES; 14];
        pits[P1_STORE] = 0;
        pits[P2_STORE] = 0;
        Self { pits, current: 0 }
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
        let range = if self.current == 0 {
            0..PITS
        } else {
            7..7 + PITS
        };
        range
            .filter(|&i| self.pits[i] > 0)
            .map(|i| i.to_string())
            .collect()
    }

    fn make_move(&mut self, mov: &str) {
        let pit: usize = mov.parse().unwrap();
        let last_pos = sow(&mut self.pits, pit, self.current);

        let own_store = if self.current == 0 {
            P1_STORE
        } else {
            P2_STORE
        };
        if last_pos == own_store {
            collect_remaining(&mut self.pits);
            return;
        }

        maybe_capture(&mut self.pits, last_pos, self.current);
        collect_remaining(&mut self.pits);

        self.current = 1 - self.current;
    }
}

// --- Dynamic evaluator: uniform priors, value 0.0 ---

struct DynMancalaEval;

impl EvalCallbacks for DynMancalaEval {
    fn evaluate(&self, _state: &dyn GameCallbacks, moves: &[String]) -> (Vec<f64>, f64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        (priors, 0.0)
    }
}

// ===========================================================================
// Task 4: Native CountingGame (GameState) — trivial game baseline
// ===========================================================================

#[derive(Clone)]
struct CountingGame(i64);

#[derive(Clone, Debug, PartialEq)]
enum CountingMove {
    Add,
    Sub,
}

impl std::fmt::Display for CountingMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    fn current_player(&self) {}

    fn available_moves(&self) -> Vec<CountingMove> {
        if self.0 == 20 {
            vec![]
        } else {
            vec![CountingMove::Add, CountingMove::Sub]
        }
    }

    fn make_move(&mut self, mov: &CountingMove) {
        match *mov {
            CountingMove::Add => self.0 += 1,
            CountingMove::Sub => self.0 -= 1,
        }
    }
}

// --- Native CountingGame evaluator: uniform priors, value 0 ---

struct CountingEval;

impl Evaluator<NativeCountingMCTS> for CountingEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        _state: &CountingGame,
        moves: &Vec<CountingMove>,
        _handle: Option<SearchHandle<NativeCountingMCTS>>,
    ) -> (Vec<f64>, i64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        (priors, 0)
    }

    fn evaluate_existing_state(
        &self,
        _state: &CountingGame,
        evaln: &i64,
        _handle: SearchHandle<NativeCountingMCTS>,
    ) -> i64 {
        *evaln
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
        *evaln
    }
}

// --- Native CountingGame MCTS config ---

#[derive(Default)]
struct NativeCountingMCTS;

impl MCTS for NativeCountingMCTS {
    type State = CountingGame;
    type Eval = CountingEval;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type TranspositionTable = ();
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::Ignore
    }

    fn fpu_value(&self) -> f64 {
        0.0
    }
}

// ===========================================================================
// Task 4: Dynamic CountingGame (GameCallbacks)
// ===========================================================================

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
        if self.0 == 20 {
            vec![]
        } else {
            vec!["Add".to_string(), "Sub".to_string()]
        }
    }

    fn make_move(&mut self, mov: &str) {
        match mov {
            "Add" => self.0 += 1,
            "Sub" => self.0 -= 1,
            _ => panic!("Unknown move: {}", mov),
        }
    }
}

// --- Dynamic CountingGame evaluator: uniform priors, value 0.0 ---

struct DynCountingEval;

impl EvalCallbacks for DynCountingEval {
    fn evaluate(&self, _state: &dyn GameCallbacks, moves: &[String]) -> (Vec<f64>, f64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        (priors, 0.0)
    }
}

// ===========================================================================
// Benchmarks
// ===========================================================================

fn bench_counting_native(c: &mut Criterion) {
    c.bench_function("counting native 10k playouts", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                CountingGame(0),
                NativeCountingMCTS,
                CountingEval,
                AlphaGoPolicy::new(1.5),
                (),
            );
            mcts.playout_n(10_000);
        });
    });
}

fn bench_counting_dynamic(c: &mut Criterion) {
    c.bench_function("counting dynamic 10k playouts", |b| {
        b.iter(|| {
            let config = DynConfig {
                exploration_constant: 1.5,
                fpu_value: 0.0,
                ..DynConfig::default()
            };
            let mut mgr = DynMCTSManager::new(
                Box::new(DynCountingGame(0)),
                Box::new(DynCountingEval),
                config,
            );
            mgr.playout_n(10_000);
        });
    });
}

fn bench_mancala_native(c: &mut Criterion) {
    c.bench_function("mancala native 10k playouts", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                Mancala::new(),
                NativeMancalaMCTS,
                MancalaEval,
                AlphaGoPolicy::new(1.5),
                (),
            );
            mcts.playout_n(10_000);
        });
    });
}

fn bench_mancala_dynamic(c: &mut Criterion) {
    c.bench_function("mancala dynamic 10k playouts", |b| {
        b.iter(|| {
            let config = DynConfig {
                exploration_constant: 1.5,
                fpu_value: 0.0,
                ..DynConfig::default()
            };
            let mut mgr = DynMCTSManager::new(
                Box::new(DynMancala::new()),
                Box::new(DynMancalaEval),
                config,
            );
            mgr.playout_n(10_000);
        });
    });
}

fn bench_mancala_native_parallel(c: &mut Criterion) {
    c.bench_function("mancala native 10k 4-thread", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                Mancala::new(),
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
            let config = DynConfig {
                exploration_constant: 1.5,
                fpu_value: 0.0,
                ..DynConfig::default()
            };
            let mut mgr = DynMCTSManager::new(
                Box::new(DynMancala::new()),
                Box::new(DynMancalaEval),
                config,
            );
            mgr.playout_n_parallel(10_000, 4);
        });
    });
}

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
