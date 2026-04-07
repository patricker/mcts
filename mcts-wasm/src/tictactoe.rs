use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Generalized M,N,K game: M cols x N rows, K in a row to win ---

const MAX_COLS: usize = 10;
const MAX_ROWS: usize = 10;
const MAX_CELLS: usize = MAX_COLS * MAX_ROWS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Empty,
    X,
    O,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Player {
    X,
    O,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TttMove(u8);

impl std::fmt::Display for TttMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug)]
struct TicTacToe {
    board: [Cell; MAX_CELLS],
    current: Player,
    cols: usize,
    rows: usize,
    k: usize, // k-in-a-row to win
}

impl TicTacToe {
    fn new(cols: usize, rows: usize, k: usize) -> Self {
        Self {
            board: [Cell::Empty; MAX_CELLS],
            current: Player::X,
            cols,
            rows,
            k,
        }
    }

    fn idx(&self, row: usize, col: usize) -> usize {
        row * self.cols + col
    }

    fn cell_count(&self) -> usize {
        self.cols * self.rows
    }

    fn winner(&self) -> Option<Player> {
        let dirs: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let cell = self.board[self.idx(r, c)];
                if cell == Cell::Empty {
                    continue;
                }
                for &(dr, dc) in &dirs {
                    let mut count = 1;
                    for step in 1..self.k {
                        let nr = r as i32 + dr * step as i32;
                        let nc = c as i32 + dc * step as i32;
                        if nr < 0
                            || nr >= self.rows as i32
                            || nc < 0
                            || nc >= self.cols as i32
                        {
                            break;
                        }
                        if self.board[self.idx(nr as usize, nc as usize)] == cell {
                            count += 1;
                        } else {
                            break;
                        }
                    }
                    if count >= self.k {
                        return match cell {
                            Cell::X => Some(Player::X),
                            Cell::O => Some(Player::O),
                            Cell::Empty => unreachable!(),
                        };
                    }
                }
            }
        }
        None
    }

    fn board_full(&self) -> bool {
        (0..self.cell_count()).all(|i| self.board[i] != Cell::Empty)
    }

    fn result_str(&self) -> &'static str {
        if let Some(w) = self.winner() {
            match w {
                Player::X => "X",
                Player::O => "O",
            }
        } else if self.board_full() {
            "Draw"
        } else {
            ""
        }
    }

    fn board_string(&self) -> String {
        (0..self.cell_count())
            .map(|i| match self.board[i] {
                Cell::Empty => ' ',
                Cell::X => 'X',
                Cell::O => 'O',
            })
            .collect()
    }
}

impl GameState for TicTacToe {
    type Move = TttMove;
    type Player = Player;
    type MoveList = Vec<TttMove>;

    fn current_player(&self) -> Player {
        self.current
    }

    fn available_moves(&self) -> Vec<TttMove> {
        if self.winner().is_some() {
            return vec![];
        }
        (0..self.cell_count())
            .filter(|&i| self.board[i] == Cell::Empty)
            .map(|i| TttMove(i as u8))
            .collect()
    }

    fn make_move(&mut self, mov: &TttMove) {
        let cell = match self.current {
            Player::X => Cell::X,
            Player::O => Cell::O,
        };
        self.board[mov.0 as usize] = cell;
        self.current = match self.current {
            Player::X => Player::O,
            Player::O => Player::X,
        };
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if self.winner().is_some() {
            Some(ProvenValue::Loss) // winner just moved, current player lost
        } else if self.board_full() {
            Some(ProvenValue::Draw)
        } else {
            None
        }
    }
}

// --- Evaluator ---

struct TttEval;

impl Evaluator<TttConfig> for TttEval {
    type StateEvaluation = ();

    fn evaluate_new_state(
        &self,
        _state: &TicTacToe,
        moves: &Vec<TttMove>,
        _: Option<SearchHandle<TttConfig>>,
    ) -> (Vec<()>, ()) {
        (vec![(); moves.len()], ())
    }

    fn interpret_evaluation_for_player(&self, _evaln: &(), _player: &Player) -> i64 {
        0
    }

    fn evaluate_existing_state(
        &self,
        _: &TicTacToe,
        _evaln: &(),
        _: SearchHandle<TttConfig>,
    ) {
    }
}

// --- MCTS Config ---

#[derive(Default)]
struct TttConfig;

impl MCTS for TttConfig {
    type State = TicTacToe;
    type Eval = TttEval;
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
pub struct TicTacToeWasm {
    manager: MCTSManager<TttConfig>,
    cols: usize,
    rows: usize,
    k: usize,
}

impl Default for TicTacToeWasm {
    fn default() -> Self {
        Self::create(3, 3, 3)
    }
}

#[wasm_bindgen]
impl TicTacToeWasm {
    fn create(cols: usize, rows: usize, k: usize) -> Self {
        let cols = cols.clamp(2, MAX_COLS);
        let rows = rows.clamp(2, MAX_ROWS);
        let k = k.clamp(2, cols.max(rows));
        Self {
            manager: MCTSManager::new(
                TicTacToe::new(cols, rows, k),
                TttConfig,
                TttEval,
                UCTPolicy::new(1.4),
                (),
            ),
            cols,
            rows,
            k,
        }
    }

    #[wasm_bindgen(constructor)]
    pub fn new(cols: u32, rows: u32, k: u32) -> Self {
        Self::create(cols as usize, rows as usize, k as usize)
    }

    pub fn cols(&self) -> u32 {
        self.cols as u32
    }

    pub fn rows(&self) -> u32 {
        self.rows as u32
    }

    pub fn win_length(&self) -> u32 {
        self.k as u32
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
            types::export_tree::<TttConfig>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    /// Board as string of length cols*rows: ' '=empty, 'X', 'O'. Row-major, top to bottom.
    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    pub fn current_player(&self) -> String {
        match self.manager.tree().root_state().current {
            Player::X => "X".into(),
            Player::O => "O".into(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        let state = self.manager.tree().root_state();
        state.winner().is_some() || state.board_full()
    }

    pub fn result(&self) -> String {
        self.manager.tree().root_state().result_str().into()
    }

    pub fn root_proven_value(&self) -> String {
        format!("{:?}", self.manager.root_proven_value())
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    /// Apply a move by cell index and advance the tree.
    pub fn apply_move(&mut self, mov: &str) -> bool {
        let idx: u8 = match mov.parse() {
            Ok(v) if (v as usize) < self.cols * self.rows => v,
            _ => return false,
        };
        let m = TttMove(idx);
        if self.manager.advance(&m).is_ok() {
            return true;
        }
        self.manager.playout_n(100);
        self.manager.advance(&m).is_ok()
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            TicTacToe::new(self.cols, self.rows, self.k),
            TttConfig,
            TttEval,
            UCTPolicy::new(1.4),
            (),
        );
    }
}
