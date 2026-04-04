//! MCTS-Solver: proving game-theoretic wins and losses.
//!
//! Plays Nim (take 1 or 2 from a pile, last stone wins) and uses the solver
//! to prove positions as won or lost — no heuristic needed, just search.
//!
//! Run: cargo run --example nim_solver

use mcts::tree_policy::*;
use mcts::*;

// --- Game ---

#[derive(Clone, Debug)]
struct Nim {
    stones: u8,
    current: Player,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Player {
    P1,
    P2,
}

#[derive(Clone, Debug, PartialEq)]
enum NimMove {
    Take1,
    Take2,
}

impl std::fmt::Display for NimMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NimMove::Take1 => write!(f, "Take 1"),
            NimMove::Take2 => write!(f, "Take 2"),
        }
    }
}

impl GameState for Nim {
    type Move = NimMove;
    type Player = Player;
    type MoveList = Vec<NimMove>;

    fn current_player(&self) -> Player {
        self.current
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
        self.current = match self.current {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        };
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if self.stones == 0 {
            // Previous player took the last stone and won.
            // Current player (who can't move) lost.
            Some(ProvenValue::Loss)
        } else {
            None
        }
    }
}

// --- Evaluator ---

struct NimEval;

impl Evaluator<NimMCTS> for NimEval {
    type StateEvaluation = Option<Player>;

    fn evaluate_new_state(
        &self,
        state: &Nim,
        moves: &Vec<NimMove>,
        _: Option<SearchHandle<NimMCTS>>,
    ) -> (Vec<()>, Option<Player>) {
        let winner = if state.stones == 0 {
            Some(match state.current {
                Player::P1 => Player::P2,
                Player::P2 => Player::P1,
            })
        } else {
            None
        };
        (vec![(); moves.len()], winner)
    }

    fn interpret_evaluation_for_player(&self, winner: &Option<Player>, player: &Player) -> i64 {
        match winner {
            Some(w) if w == player => 100,
            Some(_) => -100,
            None => 0,
        }
    }

    fn evaluate_existing_state(
        &self,
        _: &Nim,
        evaln: &Option<Player>,
        _: SearchHandle<NimMCTS>,
    ) -> Option<Player> {
        *evaln
    }
}

// --- MCTS config ---

#[derive(Default)]
struct NimMCTS;

impl MCTS for NimMCTS {
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

fn main() {
    println!("=== MCTS-Solver: Nim ===\n");
    println!("Rules: take 1 or 2 stones. Last stone wins.");
    println!("Theory: position is losing iff stones % 3 == 0.\n");

    for stones in 1u8..=9 {
        let mut mcts = MCTSManager::new(
            Nim {
                stones,
                current: Player::P1,
            },
            NimMCTS,
            NimEval,
            UCTPolicy::new(1.0),
            (),
        );
        mcts.playout_n(500);

        let proven = mcts.root_proven_value();
        let theory = if stones % 3 == 0 { "Loss" } else { "Win " };
        let best = mcts
            .best_move()
            .map(|m| format!("{m}"))
            .unwrap_or_else(|| "-".into());
        let nodes = mcts.tree().num_nodes();

        println!(
            "Stones={stones}  Proven={proven:?}  Theory={theory}  Best={best:6}  Nodes={nodes}"
        );
    }
}
