use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Game (mirrors examples/alphazero_basics.rs) ---

#[derive(Clone, Debug, PartialEq)]
struct PriorGame {
    depth: u8,
    score: i64,
}

#[derive(Clone, Debug, PartialEq)]
enum PriorMove {
    A,
    B,
    C,
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

    fn current_player(&self) -> () {}

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

// --- UCT evaluator (no priors) ---

struct UctEval;

impl Evaluator<UctConfig> for UctEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &PriorGame,
        moves: &Vec<PriorMove>,
        _: Option<SearchHandle<UctConfig>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], state.score)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self,
        _: &PriorGame,
        evaln: &i64,
        _: SearchHandle<UctConfig>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct UctConfig;

impl MCTS for UctConfig {
    type State = PriorGame;
    type Eval = UctEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();
}

// --- PUCT evaluator (with misleading priors) ---

struct PuctEval;

impl Evaluator<PuctConfig> for PuctEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &PriorGame,
        moves: &Vec<PriorMove>,
        _: Option<SearchHandle<PuctConfig>>,
    ) -> (Vec<f64>, i64) {
        let priors = if moves.len() == 3 {
            vec![0.1, 0.2, 0.7] // A=0.1, B=0.2, C=0.7 (intentionally wrong)
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
        _: SearchHandle<PuctConfig>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct PuctConfig;

impl MCTS for PuctConfig {
    type State = PriorGame;
    type Eval = PuctEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ();

    fn rng_seed(&self) -> Option<u64> {
        Some(42)
    }
}

// --- WASM API ---

#[wasm_bindgen]
pub struct PriorGameUctWasm {
    manager: MCTSManager<UctConfig>,
}

#[wasm_bindgen]
impl PriorGameUctWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(exploration_constant: f64) -> Self {
        let c = if exploration_constant > 0.0 {
            exploration_constant
        } else {
            1.5
        };
        Self {
            manager: MCTSManager::new(
                PriorGame { depth: 0, score: 0 },
                UctConfig,
                UctEval,
                UCTPolicy::new(c),
                (),
            ),
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
            types::export_tree::<UctConfig>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn reset(&mut self, exploration_constant: f64) {
        let c = if exploration_constant > 0.0 {
            exploration_constant
        } else {
            1.5
        };
        self.manager = MCTSManager::new(
            PriorGame { depth: 0, score: 0 },
            UctConfig,
            UctEval,
            UCTPolicy::new(c),
            (),
        );
    }
}

#[wasm_bindgen]
pub struct PriorGamePuctWasm {
    manager: MCTSManager<PuctConfig>,
}

#[wasm_bindgen]
impl PriorGamePuctWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(exploration_constant: f64) -> Self {
        let c = if exploration_constant > 0.0 {
            exploration_constant
        } else {
            1.5
        };
        Self {
            manager: MCTSManager::new(
                PriorGame { depth: 0, score: 0 },
                PuctConfig,
                PuctEval,
                AlphaGoPolicy::new(c),
                (),
            ),
        }
    }

    pub fn playout_n(&mut self, n: u32) {
        self.manager.playout_n(n as u64);
    }

    pub fn get_stats(&self) -> JsValue {
        let stats = types::build_stats(&self.manager, |prior| Some(*prior));
        serde_wasm_bindgen::to_value(&stats).unwrap()
    }

    pub fn get_tree(&self, max_depth: u32) -> JsValue {
        let tree =
            types::export_tree::<PuctConfig>(self.manager.tree().root_node(), max_depth, &|p| {
                Some(*p)
            });
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn reset(&mut self, exploration_constant: f64) {
        let c = if exploration_constant > 0.0 {
            exploration_constant
        } else {
            1.5
        };
        self.manager = MCTSManager::new(
            PriorGame { depth: 0, score: 0 },
            PuctConfig,
            PuctEval,
            AlphaGoPolicy::new(c),
            (),
        );
    }
}
