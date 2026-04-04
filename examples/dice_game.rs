//! Chance nodes: MCTS with stochastic transitions.
//!
//! A dice game where the player chooses Roll or Stop each turn.
//! After Roll, a d6 is added to the score. Game ends at score >= 20.
//! MCTS correctly learns that rolling is always better than stopping.
//!
//! Run: cargo run --example dice_game

use mcts::tree_policy::*;
use mcts::*;

// --- Game ---

#[derive(Clone, Debug)]
struct DiceGame {
    score: i64,
    pending_roll: bool,
    stopped: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum DiceMove {
    Roll,
    Stop,
    Die(u8), // chance outcome: 1-6
}

impl std::fmt::Display for DiceMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DiceMove::Roll => write!(f, "Roll"),
            DiceMove::Stop => write!(f, "Stop"),
            DiceMove::Die(n) => write!(f, "Die({n})"),
        }
    }
}

impl GameState for DiceGame {
    type Move = DiceMove;
    type Player = ();
    type MoveList = Vec<DiceMove>;

    fn current_player(&self) -> () {}

    fn available_moves(&self) -> Vec<DiceMove> {
        if self.pending_roll || self.stopped || self.score >= 20 {
            vec![]
        } else {
            vec![DiceMove::Roll, DiceMove::Stop]
        }
    }

    fn make_move(&mut self, mov: &DiceMove) {
        match mov {
            DiceMove::Roll => self.pending_roll = true,
            DiceMove::Stop => self.stopped = true,
            DiceMove::Die(v) => {
                self.score += *v as i64;
                self.pending_roll = false;
            }
        }
    }

    fn chance_outcomes(&self) -> Option<Vec<(DiceMove, f64)>> {
        if self.pending_roll {
            Some(
                (1..=6)
                    .map(|i| (DiceMove::Die(i), 1.0 / 6.0))
                    .collect(),
            )
        } else {
            None
        }
    }
}

// --- Evaluator ---

struct DiceEval;

impl Evaluator<DiceMCTS> for DiceEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &DiceGame,
        moves: &Vec<DiceMove>,
        _: Option<SearchHandle<DiceMCTS>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], state.score)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self,
        state: &DiceGame,
        _evaln: &i64,
        _: SearchHandle<DiceMCTS>,
    ) -> i64 {
        // Re-evaluate from current state: open-loop MCTS means different
        // chance outcomes land on the same tree node.
        state.score
    }
}

// --- MCTS config ---

#[derive(Default)]
struct DiceMCTS;

impl MCTS for DiceMCTS {
    type State = DiceGame;
    type Eval = DiceEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();
}

fn main() {
    println!("=== Chance Nodes: Dice Game ===\n");
    println!("Rules: Roll a d6 to add to score, or Stop. Terminal at score >= 20.");
    println!("Optimal strategy: always Roll (E[d6] = 3.5 > 0).\n");

    for start_score in [0, 5, 10, 15, 18] {
        let mut mcts = MCTSManager::new(
            DiceGame {
                score: start_score,
                pending_roll: false,
                stopped: false,
            },
            DiceMCTS,
            DiceEval,
            UCTPolicy::new(0.5),
            (),
        );
        mcts.playout_n(50_000);

        let best = mcts
            .best_move()
            .map(|m| format!("{m}"))
            .unwrap_or_else(|| "terminal".into());

        let stats = mcts.root_child_stats();
        let roll_stats = stats.iter().find(|s| s.mov == DiceMove::Roll);
        let stop_stats = stats.iter().find(|s| s.mov == DiceMove::Stop);

        print!("Score={start_score:2}  Best={best:8}");
        if let (Some(r), Some(s)) = (roll_stats, stop_stats) {
            print!(
                "  Roll: {:.1} avg ({} visits)  Stop: {:.1} avg ({} visits)",
                r.avg_reward, r.visits, s.avg_reward, s.visits
            );
        }
        println!();
    }
}
