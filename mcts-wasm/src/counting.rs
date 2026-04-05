use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Game (mirrors examples/counting_game.rs) ---

#[derive(Clone)]
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

    fn current_player(&self) -> () {}

    fn available_moves(&self) -> Vec<Move> {
        if self.0 == 100 {
            vec![]
        } else {
            vec![Move::Add, Move::Sub]
        }
    }

    fn make_move(&mut self, mov: &Move) {
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

struct Eval;

impl Evaluator<Config> for Eval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &CountingGame,
        moves: &Vec<Move>,
        _: Option<SearchHandle<Config>>,
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
        _: SearchHandle<Config>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct Config;

impl MCTS for Config {
    type State = CountingGame;
    type Eval = Eval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn virtual_loss(&self) -> i64 {
        500
    }

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}

// --- WASM API ---

#[wasm_bindgen]
pub struct CountingGameWasm {
    manager: MCTSManager<Config>,
}

#[wasm_bindgen]
impl CountingGameWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(exploration_constant: f64) -> Self {
        let c = if exploration_constant > 0.0 {
            exploration_constant
        } else {
            1.0
        };
        Self {
            manager: MCTSManager::new(
                CountingGame(0),
                Config,
                Eval,
                UCTPolicy::new(c),
                ApproxTable::new(1024),
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
            types::export_tree::<Config>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn num_nodes(&self) -> usize {
        self.manager.tree().num_nodes()
    }

    pub fn reset(&mut self, exploration_constant: f64) {
        let c = if exploration_constant > 0.0 {
            exploration_constant
        } else {
            1.0
        };
        self.manager = MCTSManager::new(
            CountingGame(0),
            Config,
            Eval,
            UCTPolicy::new(c),
            ApproxTable::new(1024),
        );
    }
}
