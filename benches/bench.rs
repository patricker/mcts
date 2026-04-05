use criterion::{criterion_group, criterion_main, Criterion};
use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;

#[derive(Clone)]
struct CountingGame(i64);

#[derive(Clone, Debug)]
enum Move {
    Add,
    Sub,
}

impl GameState for CountingGame {
    type Move = Move;
    type Player = ();
    type MoveList = Vec<Move>;

    fn current_player(&self) -> Self::Player {}
    fn available_moves(&self) -> Vec<Move> {
        if self.0 == 100 {
            vec![]
        } else {
            vec![Move::Add, Move::Sub]
        }
    }
    fn make_move(&mut self, mov: &Self::Move) {
        match *mov {
            Move::Add => self.0 += 1,
            Move::Sub => self.0 -= 1,
        }
    }
}

impl TranspositionHash for CountingGame {
    fn hash(&self) -> u64 {
        self.0 as u64
    }
}

struct MyEvaluator;

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &CountingGame,
        moves: &Vec<Move>,
        _: Option<SearchHandle<MyMCTS>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], state.0)
    }
    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }
    fn evaluate_existing_state(
        &self,
        _: &CountingGame,
        evaln: &i64,
        _: SearchHandle<MyMCTS>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = CountingGame;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}

fn bench_single_threaded(c: &mut Criterion) {
    c.bench_function("playout_n 100k single-threaded", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                CountingGame(0),
                MyMCTS,
                MyEvaluator,
                UCTPolicy::new(5.0),
                ApproxTable::new(1024),
            );
            mcts.playout_n(100_000);
        });
    });
}

fn bench_parallel(c: &mut Criterion) {
    c.bench_function("playout_n 100k 4-thread", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                CountingGame(0),
                MyMCTS,
                MyEvaluator,
                UCTPolicy::new(5.0),
                ApproxTable::new(1024),
            );
            mcts.playout_n_parallel(100_000, 4);
        });
    });
}

// --- PUCT benchmark (AlphaGoPolicy with priors) ---

struct PuctEval;

impl Evaluator<PuctMCTS> for PuctEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &CountingGame,
        moves: &Vec<Move>,
        _: Option<SearchHandle<PuctMCTS>>,
    ) -> (Vec<f64>, i64) {
        let n = moves.len();
        let priors = if n > 0 {
            vec![1.0 / n as f64; n]
        } else {
            vec![]
        };
        (priors, state.0)
    }
    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }
    fn evaluate_existing_state(
        &self,
        _: &CountingGame,
        evaln: &i64,
        _: SearchHandle<PuctMCTS>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct PuctMCTS;

impl MCTS for PuctMCTS {
    type State = CountingGame;
    type Eval = PuctEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
    fn fpu_value(&self) -> f64 {
        0.0
    }
}

fn bench_puct(c: &mut Criterion) {
    c.bench_function("playout_n 100k PUCT", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                CountingGame(0),
                PuctMCTS,
                PuctEval,
                AlphaGoPolicy::new(1.5),
                (),
            );
            mcts.playout_n(100_000);
        });
    });
}

// --- Solver benchmark (Nim with solver) ---

#[derive(Clone, PartialEq)]
struct Nim {
    stones: u32,
}

#[derive(Clone, Debug, PartialEq)]
enum NimMove {
    Take1,
    Take2,
}

impl std::fmt::Display for NimMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NimMove::Take1 => write!(f, "Take1"),
            NimMove::Take2 => write!(f, "Take2"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
enum NimPlayer {
    P1,
    P2,
}

impl GameState for Nim {
    type Move = NimMove;
    type Player = NimPlayer;
    type MoveList = Vec<NimMove>;

    fn current_player(&self) -> NimPlayer {
        NimPlayer::P1 // simplified
    }

    fn available_moves(&self) -> Vec<NimMove> {
        match self.stones {
            0 => vec![],
            1 => vec![NimMove::Take1],
            _ => vec![NimMove::Take1, NimMove::Take2],
        }
    }

    fn make_move(&mut self, mov: &NimMove) {
        match mov {
            NimMove::Take1 => self.stones -= 1,
            NimMove::Take2 => self.stones -= 2,
        }
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if self.stones == 0 {
            Some(ProvenValue::Win)
        } else {
            None
        }
    }
}

struct NimEval;

impl Evaluator<SolverMCTS> for NimEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        _: &Nim,
        moves: &Vec<NimMove>,
        _: Option<SearchHandle<SolverMCTS>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], 0)
    }
    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &NimPlayer) -> i64 {
        *evaln
    }
    fn evaluate_existing_state(&self, _: &Nim, evaln: &i64, _: SearchHandle<SolverMCTS>) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct SolverMCTS;

impl MCTS for SolverMCTS {
    type State = Nim;
    type Eval = NimEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();

    fn solver_enabled(&self) -> bool {
        true
    }
}

fn bench_solver(c: &mut Criterion) {
    c.bench_function("playout_n 50k solver (Nim 10)", |b| {
        b.iter(|| {
            let mut mcts = MCTSManager::new(
                Nim { stones: 10 },
                SolverMCTS,
                NimEval,
                UCTPolicy::new(1.0),
                (),
            );
            mcts.playout_n(50_000);
        });
    });
}

criterion_group!(
    benches,
    bench_single_threaded,
    bench_parallel,
    bench_puct,
    bench_solver
);
criterion_main!(benches);
