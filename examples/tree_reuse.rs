//! Tree re-rooting: preserving search across turns.
//!
//! Shows how `advance()` commits to a move and re-roots the search tree,
//! preserving the subtree below the chosen move. This avoids re-searching
//! positions that were already explored — critical for pondering (searching
//! during the opponent's turn).
//!
//! Run: cargo run --example tree_reuse
//! Output: cargo run --example tree_reuse > examples/output/tree_reuse.txt

use mcts::tree_policy::*;
use mcts::*;

// --- Reuse the CountingGame from the main example ---

#[derive(Clone, Debug, PartialEq)]
struct CountingGame(i64);

#[derive(Clone, Debug, PartialEq)]
enum Move {
    Add,
    Sub,
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Move::Add => write!(f, "Add"),
            Move::Sub => write!(f, "Sub"),
        }
    }
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
    fn make_move(&mut self, mov: &Move) {
        match mov {
            Move::Add => self.0 += 1,
            Move::Sub => self.0 -= 1,
        }
    }
}

struct MyEval;

impl Evaluator<MyMCTS> for MyEval {
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
    type Eval = MyEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();
}

// region: reuse_loop
fn main() {
    println!("=== Tree Re-Rooting ===\n");

    let mut mcts = MCTSManager::new(CountingGame(0), MyMCTS, MyEval, UCTPolicy::new(0.5), ());

    // Simulate 5 turns of play, reusing the search tree each turn
    for turn in 1..=5 {
        // Search from current position
        mcts.playout_n_parallel(10_000, 4);

        let root_state = mcts.tree().root_state().0;
        let best = mcts.best_move().unwrap();
        let nodes = mcts.tree().num_nodes();

        println!("Turn {turn}: state={root_state}, best={best}, nodes={nodes}");

        // Show child stats
        let stats = mcts.root_child_stats();
        for s in &stats {
            println!(
                "  {}: {} visits, {:.1} avg reward",
                s.mov, s.visits, s.avg_reward
            );
        }

        // Commit to the best move and re-root
        mcts.advance(&best).unwrap();

        let new_nodes = mcts.tree().num_nodes();
        println!(
            "  -> Advanced to state={}, preserved {new_nodes} nodes\n",
            mcts.tree().root_state().0
        );
    }
}
// endregion: reuse_loop
