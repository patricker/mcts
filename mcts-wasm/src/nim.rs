use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Game (mirrors examples/nim_solver.rs) ---

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
            NimMove::Take1 => write!(f, "Take1"),
            NimMove::Take2 => write!(f, "Take2"),
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
            Some(ProvenValue::Loss)
        } else {
            None
        }
    }
}

struct NimEval;

impl Evaluator<NimConfig> for NimEval {
    type StateEvaluation = Option<Player>;

    fn evaluate_new_state(
        &self,
        state: &Nim,
        moves: &Vec<NimMove>,
        _: Option<SearchHandle<NimConfig>>,
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
        _: SearchHandle<NimConfig>,
    ) -> Option<Player> {
        *evaln
    }
}

#[derive(Default)]
struct NimConfig;

impl MCTS for NimConfig {
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

// --- WASM API ---

#[wasm_bindgen]
pub struct NimWasm {
    manager: MCTSManager<NimConfig>,
    initial_stones: u8,
}

#[wasm_bindgen]
impl NimWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(stones: u8) -> Self {
        let s = if stones > 0 { stones } else { 5 };
        Self {
            manager: MCTSManager::new(
                Nim {
                    stones: s,
                    current: Player::P1,
                },
                NimConfig,
                NimEval,
                UCTPolicy::new(1.0),
                (),
            ),
            initial_stones: s,
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
            types::export_tree::<NimConfig>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn root_proven_value(&self) -> String {
        format!("{:?}", self.manager.root_proven_value())
    }

    pub fn current_stones(&self) -> u8 {
        self.manager.tree().root_state().stones
    }

    pub fn current_player(&self) -> String {
        format!("{:?}", self.manager.tree().root_state().current)
    }

    pub fn is_terminal(&self) -> bool {
        self.manager.tree().root_state().stones == 0
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    /// Apply a move and advance the tree (preserving search).
    pub fn apply_move(&mut self, mov: &str) -> bool {
        let m = match mov {
            "Take1" => NimMove::Take1,
            "Take2" => NimMove::Take2,
            _ => return false,
        };
        self.manager.advance(&m).is_ok()
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            Nim {
                stones: self.initial_stones,
                current: Player::P1,
            },
            NimConfig,
            NimEval,
            UCTPolicy::new(1.0),
            (),
        );
    }
}
