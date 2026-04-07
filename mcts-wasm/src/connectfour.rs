use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Game ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Empty,
    Red,
    Yellow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Player {
    Red,
    Yellow,
}

#[derive(Clone, Debug, PartialEq)]
struct CfMove(u8);

impl std::fmt::Display for CfMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

const COLS: usize = 7;
const ROWS: usize = 6;

#[derive(Clone, Debug)]
struct ConnectFour {
    board: [[Cell; COLS]; ROWS],
    current: Player,
}

impl ConnectFour {
    fn new() -> Self {
        Self {
            board: [[Cell::Empty; COLS]; ROWS],
            current: Player::Red,
        }
    }

    fn cell_for_player(player: Player) -> Cell {
        match player {
            Player::Red => Cell::Red,
            Player::Yellow => Cell::Yellow,
        }
    }

    /// Find the lowest empty row in a column, or None if full.
    fn drop_row(&self, col: usize) -> Option<usize> {
        (0..ROWS).find(|&row| self.board[row][col] == Cell::Empty)
    }

    /// Check if the given cell color has 4 in a row through (row, col).
    fn check_win_at(&self, row: usize, col: usize, cell: Cell) -> bool {
        if self.board[row][col] != cell {
            return false;
        }
        // Directions: horizontal, vertical, diagonal /, diagonal \.
        let directions: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
        for (dr, dc) in directions {
            let mut count = 1;
            // Positive direction
            for i in 1..4 {
                let r = row as i32 + dr * i;
                let c = col as i32 + dc * i;
                if r < 0 || r >= ROWS as i32 || c < 0 || c >= COLS as i32 {
                    break;
                }
                if self.board[r as usize][c as usize] != cell {
                    break;
                }
                count += 1;
            }
            // Negative direction
            for i in 1..4 {
                let r = row as i32 - dr * i;
                let c = col as i32 - dc * i;
                if r < 0 || r >= ROWS as i32 || c < 0 || c >= COLS as i32 {
                    break;
                }
                if self.board[r as usize][c as usize] != cell {
                    break;
                }
                count += 1;
            }
            if count >= 4 {
                return true;
            }
        }
        false
    }

    /// Check if the player who just moved won.
    fn has_winner(&self) -> Option<Player> {
        for row in 0..ROWS {
            for col in 0..COLS {
                match self.board[row][col] {
                    Cell::Red if self.check_win_at(row, col, Cell::Red) => return Some(Player::Red),
                    Cell::Yellow if self.check_win_at(row, col, Cell::Yellow) => {
                        return Some(Player::Yellow)
                    }
                    _ => {}
                }
            }
        }
        None
    }

    fn is_full(&self) -> bool {
        (0..COLS).all(|col| self.board[ROWS - 1][col] != Cell::Empty)
    }

    fn board_string(&self) -> String {
        let mut s = String::with_capacity(ROWS * COLS);
        // Top row first (row 5 down to row 0)
        for row in (0..ROWS).rev() {
            for col in 0..COLS {
                s.push(match self.board[row][col] {
                    Cell::Empty => ' ',
                    Cell::Red => 'R',
                    Cell::Yellow => 'Y',
                });
            }
        }
        s
    }
}

impl GameState for ConnectFour {
    type Move = CfMove;
    type Player = Player;
    type MoveList = Vec<CfMove>;

    fn current_player(&self) -> Player {
        self.current
    }

    fn available_moves(&self) -> Vec<CfMove> {
        // No moves if someone won
        if self.has_winner().is_some() {
            return vec![];
        }
        (0..COLS as u8)
            .filter(|&col| self.drop_row(col as usize).is_some())
            .map(CfMove)
            .collect()
    }

    fn make_move(&mut self, mov: &CfMove) {
        let col = mov.0 as usize;
        if let Some(row) = self.drop_row(col) {
            self.board[row][col] = Self::cell_for_player(self.current);
        }
        self.current = match self.current {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        };
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        // If someone won, the winner just moved, so current player lost.
        if self.has_winner().is_some() {
            return Some(ProvenValue::Loss);
        }
        if self.is_full() {
            return Some(ProvenValue::Draw);
        }
        None
    }
}

// --- Evaluator ---

struct CfEval;

impl CfEval {
    /// Score a window of 4 cells from one player's perspective.
    fn score_window(window: &[Cell; 4], player_cell: Cell, opp_cell: Cell) -> i64 {
        let mine = window.iter().filter(|&&c| c == player_cell).count();
        let theirs = window.iter().filter(|&&c| c == opp_cell).count();
        let empty = window.iter().filter(|&&c| c == Cell::Empty).count();

        if mine == 4 {
            return 1000;
        }
        if theirs == 4 {
            return -1000;
        }
        // Threat: 3 of mine + 1 empty
        if mine == 3 && empty == 1 {
            return 50;
        }
        // Opponent threat: 3 of theirs + 1 empty
        if theirs == 3 && empty == 1 {
            return -80; // Penalize more — blocking is urgent
        }
        if mine == 2 && empty == 2 {
            return 5;
        }
        if theirs == 2 && empty == 2 {
            return -5;
        }
        0
    }

    /// Evaluate the board from Red's perspective.
    fn evaluate_board(state: &ConnectFour) -> i64 {
        let mut score: i64 = 0;
        let red = Cell::Red;
        let yellow = Cell::Yellow;

        // Center column bonus
        for row in 0..ROWS {
            if state.board[row][3] == red {
                score += 3;
            } else if state.board[row][3] == yellow {
                score -= 3;
            }
        }

        // Score all horizontal windows of 4
        for row in 0..ROWS {
            for col in 0..COLS - 3 {
                let w = [
                    state.board[row][col],
                    state.board[row][col + 1],
                    state.board[row][col + 2],
                    state.board[row][col + 3],
                ];
                score += Self::score_window(&w, red, yellow);
            }
        }

        // Score all vertical windows of 4
        for col in 0..COLS {
            for row in 0..ROWS - 3 {
                let w = [
                    state.board[row][col],
                    state.board[row + 1][col],
                    state.board[row + 2][col],
                    state.board[row + 3][col],
                ];
                score += Self::score_window(&w, red, yellow);
            }
        }

        // Score all diagonal (up-right) windows
        for row in 0..ROWS - 3 {
            for col in 0..COLS - 3 {
                let w = [
                    state.board[row][col],
                    state.board[row + 1][col + 1],
                    state.board[row + 2][col + 2],
                    state.board[row + 3][col + 3],
                ];
                score += Self::score_window(&w, red, yellow);
            }
        }

        // Score all diagonal (down-right) windows
        for row in 3..ROWS {
            for col in 0..COLS - 3 {
                let w = [
                    state.board[row][col],
                    state.board[row - 1][col + 1],
                    state.board[row - 2][col + 2],
                    state.board[row - 3][col + 3],
                ];
                score += Self::score_window(&w, red, yellow);
            }
        }

        score
    }
}

impl Evaluator<CfConfig> for CfEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &ConnectFour,
        moves: &Vec<CfMove>,
        _: Option<SearchHandle<CfConfig>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], Self::evaluate_board(state))
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match player {
            Player::Red => *evaln,
            Player::Yellow => -*evaln,
        }
    }

    fn evaluate_existing_state(
        &self,
        state: &ConnectFour,
        _evaln: &i64,
        _: SearchHandle<CfConfig>,
    ) -> i64 {
        Self::evaluate_board(state)
    }
}

// --- MCTS Config ---

#[derive(Default)]
struct CfConfig;

impl MCTS for CfConfig {
    type State = ConnectFour;
    type Eval = CfEval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();

    fn solver_enabled(&self) -> bool {
        false
    }
}

// --- WASM API ---

#[wasm_bindgen]
pub struct ConnectFourWasm {
    manager: MCTSManager<CfConfig>,
}

impl Default for ConnectFourWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl ConnectFourWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            manager: MCTSManager::new(
                ConnectFour::new(),
                CfConfig,
                CfEval,
                UCTPolicy::new(1.4),
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
            types::export_tree::<CfConfig>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    pub fn current_player(&self) -> String {
        match self.manager.tree().root_state().current {
            Player::Red => "Red".into(),
            Player::Yellow => "Yellow".into(),
        }
    }

    pub fn is_terminal(&self) -> bool {
        self.manager
            .tree()
            .root_state()
            .terminal_value()
            .is_some()
    }

    pub fn result(&self) -> String {
        let state = self.manager.tree().root_state();
        if let Some(winner) = state.has_winner() {
            match winner {
                Player::Red => "Red".into(),
                Player::Yellow => "Yellow".into(),
            }
        } else if state.is_full() {
            "Draw".into()
        } else {
            String::new()
        }
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    /// Apply a move and advance the tree (preserving search).
    /// Runs a few playouts first if needed to ensure the child is expanded.
    pub fn apply_move(&mut self, col: &str) -> bool {
        let col_num: u8 = match col.parse() {
            Ok(n) if n < COLS as u8 => n,
            _ => return false,
        };
        let m = CfMove(col_num);
        if self.manager.advance(&m).is_ok() {
            return true;
        }
        // Child not expanded yet — run playouts to expand, then retry
        self.manager.playout_n(100);
        self.manager.advance(&m).is_ok()
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            ConnectFour::new(),
            CfConfig,
            CfEval,
            UCTPolicy::new(1.4),
            (),
        );
    }
}
