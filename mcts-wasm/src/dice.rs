use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Game (mirrors examples/dice_game.rs) ---

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
    Die(u8),
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
            Some((1..=6).map(|i| (DiceMove::Die(i), 1.0 / 6.0)).collect())
        } else {
            None
        }
    }
}

struct DiceEval;

impl Evaluator<DiceConfig> for DiceEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &DiceGame,
        moves: &Vec<DiceMove>,
        _: Option<SearchHandle<DiceConfig>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], state.score)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self,
        state: &DiceGame,
        _: &i64,
        _: SearchHandle<DiceConfig>,
    ) -> i64 {
        state.score
    }
}

#[derive(Default)]
struct DiceConfig;

impl MCTS for DiceConfig {
    type State = DiceGame;
    type Eval = DiceEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();
}

// --- WASM API ---

#[wasm_bindgen]
pub struct DiceGameWasm {
    manager: MCTSManager<DiceConfig>,
    start_score: i64,
}

#[wasm_bindgen]
impl DiceGameWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(start_score: i64) -> Self {
        Self {
            manager: MCTSManager::new(
                DiceGame {
                    score: start_score,
                    pending_roll: false,
                    stopped: false,
                },
                DiceConfig,
                DiceEval,
                UCTPolicy::new(0.5),
                (),
            ),
            start_score,
        }
    }

    pub fn playout_n(&mut self, n: u32) {
        self.manager.playout_n(n as u64);
    }

    pub fn get_stats(&self) -> JsValue {
        let stats = types::build_stats(&self.manager, |_| None);
        serde_wasm_bindgen::to_value(&stats).unwrap()
    }

    pub fn get_tree(&self, max_depth: u32) -> JsValue {
        let tree =
            types::export_tree::<DiceConfig>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn current_score(&self) -> i64 {
        self.manager.tree().root_state().score
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            DiceGame {
                score: self.start_score,
                pending_roll: false,
                stopped: false,
            },
            DiceConfig,
            DiceEval,
            UCTPolicy::new(0.5),
            (),
        );
    }
}
