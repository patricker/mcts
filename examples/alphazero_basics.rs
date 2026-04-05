//! AlphaZero-style search: neural priors, FPU, Dirichlet noise, temperature.
//!
//! Demonstrates the PUCT (AlphaGo) policy with all the knobs needed for
//! self-play training: prior probabilities, first play urgency, root noise
//! for exploration, and temperature-based move selection.
//!
//! Run: cargo run --example alphazero_basics
//! Output: cargo run --example alphazero_basics > examples/output/alphazero_basics.txt

use mcts::tree_policy::*;
use mcts::*;

// --- Game: simple 3-move game with known optimal play ---

// region: prior_game
/// A game where the player picks A, B, or C. Each has a different value.
/// The "neural network prior" is intentionally wrong (favors C),
/// but MCTS should overcome it and find A (the best move).
#[derive(Clone, Debug, PartialEq)]
struct PriorGame {
    depth: u8,
    score: i64,
}

#[derive(Clone, Debug, PartialEq)]
enum PriorMove {
    A, // best: +10
    B, // medium: +5
    C, // worst: +1 (but highest prior!)
}

impl std::fmt::Display for PriorMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PriorMove::A => write!(f, "A(+10)"),
            PriorMove::B => write!(f, "B(+5)"),
            PriorMove::C => write!(f, "C(+1)"),
        }
    }
}

impl GameState for PriorGame {
    type Move = PriorMove;
    type Player = ();
    type MoveList = Vec<PriorMove>;

    fn current_player(&self) -> Self::Player {}

    fn available_moves(&self) -> Vec<PriorMove> {
        if self.depth >= 3 {
            vec![]
        } else {
            vec![PriorMove::A, PriorMove::B, PriorMove::C]
        }
    }

    fn make_move(&mut self, mov: &PriorMove) {
        self.depth += 1;
        match mov {
            PriorMove::A => self.score += 10,
            PriorMove::B => self.score += 5,
            PriorMove::C => self.score += 1,
        }
    }
}
// endregion: prior_game

// --- Evaluator with intentionally misleading priors ---

// region: prior_evaluator
struct PriorEval;

impl Evaluator<PriorMCTS> for PriorEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &PriorGame,
        moves: &Vec<PriorMove>,
        _: Option<SearchHandle<PriorMCTS>>,
    ) -> (Vec<f64>, i64) {
        // The "neural network" thinks C is best (0.7 prior), A is worst (0.1).
        // MCTS must overcome this misleading prior through search.
        let priors = if moves.len() == 3 {
            vec![0.1, 0.2, 0.7] // A=0.1, B=0.2, C=0.7 (wrong!)
        } else {
            vec![]
        };
        (priors, state.score)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self,
        _: &PriorGame,
        evaln: &i64,
        _: SearchHandle<PriorMCTS>,
    ) -> i64 {
        *evaln
    }
}
// endregion: prior_evaluator

// --- MCTS config with AlphaZero features ---

// region: alphazero_config
#[derive(Default)]
struct PriorMCTS;

impl MCTS for PriorMCTS {
    type State = PriorGame;
    type Eval = PriorEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ();

    // Dirichlet noise: blend prior with Dir(0.3) noise at root.
    // epsilon=0.25 means 75% prior + 25% noise.
    fn dirichlet_noise(&self) -> Option<(f64, f64)> {
        Some((0.25, 0.3))
    }

    // Temperature: sample proportional to visits (exploration mode).
    fn selection_temperature(&self) -> f64 {
        1.0
    }

    // Seeded for reproducibility
    fn rng_seed(&self) -> Option<u64> {
        Some(42)
    }
}
// endregion: alphazero_config

// region: run_alphazero
fn main() {
    println!("=== AlphaZero-Style Search ===\n");
    println!("Game: pick A(+10), B(+5), or C(+1) at each step, depth 3.");
    println!("Prior: C=0.7, B=0.2, A=0.1 (intentionally wrong).");
    println!("Features: PUCT policy, Dirichlet noise, temperature=1.0\n");

    let mut mcts = MCTSManager::new(
        PriorGame { depth: 0, score: 0 },
        PriorMCTS,
        PriorEval,
        AlphaGoPolicy::new(1.5),
        (),
    );
    mcts.playout_n(10_000);

    // Show root stats
    println!("Root children after 10K playouts:");
    let stats = mcts.root_child_stats();
    for s in &stats {
        println!(
            "  {} — visits: {:5}, avg_reward: {:6.1}, prior: {:.3}",
            s.mov, s.visits, s.avg_reward, s.move_evaluation
        );
    }

    // Temperature-based selection (sample 20 moves)
    println!("\n20 temperature-sampled moves (temp=1.0):");
    let moves: Vec<_> = (0..20).map(|_| mcts.best_move().unwrap()).collect();
    let a_count = moves.iter().filter(|m| **m == PriorMove::A).count();
    let b_count = moves.iter().filter(|m| **m == PriorMove::B).count();
    let c_count = moves.iter().filter(|m| **m == PriorMove::C).count();
    println!("  A: {a_count}, B: {b_count}, C: {c_count}");
    println!("  (A should dominate despite low prior)\n");

    // Principal variation (always argmax, ignores temperature)
    let pv = mcts.principal_variation(3);
    println!(
        "Principal variation: {:?}",
        pv.iter().map(|m| format!("{m}")).collect::<Vec<_>>()
    );
}
// endregion: run_alphazero
