use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Generalized gravity game: configurable cols, rows, k-in-a-row, N players ---

const MAX_COLS: usize = 10;
const MAX_ROWS: usize = 10;
const MAX_PLAYERS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Empty,
    Player(u8), // 0-indexed player number
}

#[derive(Clone, Debug)]
struct ConnectFour {
    board: [[Cell; MAX_COLS]; MAX_ROWS],
    cols: usize,
    rows: usize,
    k: usize,
    num_players: usize,
    current: u8, // 0-indexed
}

#[derive(Clone, Debug, PartialEq)]
struct CfMove(u8); // column index

impl std::fmt::Display for CfMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ConnectFour {
    fn new(cols: usize, rows: usize, k: usize, num_players: usize) -> Self {
        Self {
            board: [[Cell::Empty; MAX_COLS]; MAX_ROWS],
            cols,
            rows,
            k,
            num_players,
            current: 0,
        }
    }

    fn drop_row(&self, col: usize) -> Option<usize> {
        (0..self.rows).find(|&row| self.board[row][col] == Cell::Empty)
    }

    fn check_win_at(&self, row: usize, col: usize, cell: Cell) -> bool {
        if self.board[row][col] != cell {
            return false;
        }
        let dirs: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
        for (dr, dc) in dirs {
            let mut count = 1usize;
            for sign in [1i32, -1i32] {
                for step in 1..self.k {
                    let r = row as i32 + dr * sign * step as i32;
                    let c = col as i32 + dc * sign * step as i32;
                    if r < 0 || r >= self.rows as i32 || c < 0 || c >= self.cols as i32 {
                        break;
                    }
                    if self.board[r as usize][c as usize] == cell {
                        count += 1;
                    } else {
                        break;
                    }
                }
            }
            if count >= self.k {
                return true;
            }
        }
        false
    }

    fn winner(&self) -> Option<u8> {
        for row in 0..self.rows {
            for col in 0..self.cols {
                if let Cell::Player(p) = self.board[row][col] {
                    if self.check_win_at(row, col, Cell::Player(p)) {
                        return Some(p);
                    }
                }
            }
        }
        None
    }

    fn is_full(&self) -> bool {
        (0..self.cols).all(|col| self.board[self.rows - 1][col] != Cell::Empty)
    }

    fn board_string(&self) -> String {
        let mut s = String::with_capacity(self.rows * self.cols);
        // Top row first
        for row in (0..self.rows).rev() {
            for col in 0..self.cols {
                s.push(match self.board[row][col] {
                    Cell::Empty => ' ',
                    Cell::Player(p) => (b'1' + p) as char, // '1', '2', '3', '4'
                });
            }
        }
        s
    }

    /// Evaluate from a specific player's perspective.
    fn evaluate_for(&self, player: u8) -> i64 {
        let mut score: i64 = 0;
        let my_cell = Cell::Player(player);

        // Center column bonus
        let center = self.cols / 2;
        for row in 0..self.rows {
            if self.board[row][center] == my_cell {
                score += 3;
            }
        }

        // Score windows in all 4 directions
        let windows: Vec<(usize, usize, i32, i32)> = {
            let mut w = Vec::new();
            // Horizontal
            if self.cols >= self.k {
                for row in 0..self.rows {
                    for col in 0..=self.cols - self.k {
                        w.push((row, col, 0, 1));
                    }
                }
            }
            // Vertical
            if self.rows >= self.k {
                for col in 0..self.cols {
                    for row in 0..=self.rows - self.k {
                        w.push((row, col, 1, 0));
                    }
                }
            }
            // Diagonal up-right
            if self.rows >= self.k && self.cols >= self.k {
                for row in 0..=self.rows - self.k {
                    for col in 0..=self.cols - self.k {
                        w.push((row, col, 1, 1));
                    }
                }
            }
            // Diagonal down-right
            if self.rows >= self.k && self.cols >= self.k {
                for row in (self.k - 1)..self.rows {
                    for col in 0..=self.cols - self.k {
                        w.push((row, col, -1, 1));
                    }
                }
            }
            w
        };

        for (row, col, dr, dc) in windows {
            let mut mine = 0usize;
            let mut empty = 0usize;
            let mut theirs = 0usize;
            for step in 0..self.k {
                let r = (row as i32 + dr * step as i32) as usize;
                let c = (col as i32 + dc * step as i32) as usize;
                let cell = self.board[r][c];
                if cell == my_cell {
                    mine += 1;
                } else if cell == Cell::Empty {
                    empty += 1;
                } else {
                    theirs += 1;
                }
            }
            if mine == self.k {
                score += 1000;
            } else if theirs == self.k {
                score -= 1000;
            } else if mine == self.k - 1 && empty == 1 {
                score += 50;
            } else if theirs == self.k - 1 && empty == 1 {
                score -= 80;
            } else if mine >= 2 && empty == self.k - mine {
                score += 5;
            }
        }

        score
    }
}

impl GameState for ConnectFour {
    type Move = CfMove;
    type Player = u8;
    type MoveList = Vec<CfMove>;

    fn current_player(&self) -> u8 {
        self.current
    }

    fn available_moves(&self) -> Vec<CfMove> {
        if self.winner().is_some() {
            return vec![];
        }
        (0..self.cols as u8)
            .filter(|&col| self.drop_row(col as usize).is_some())
            .map(CfMove)
            .collect()
    }

    fn make_move(&mut self, mov: &CfMove) {
        let col = mov.0 as usize;
        if let Some(row) = self.drop_row(col) {
            self.board[row][col] = Cell::Player(self.current);
        }
        self.current = (self.current + 1) % self.num_players as u8;
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if self.winner().is_some() {
            // Winner just moved, current player lost
            Some(ProvenValue::Loss)
        } else if self.is_full() {
            Some(ProvenValue::Draw)
        } else {
            None
        }
    }
}

// --- Evaluator ---

struct CfEval;

/// The state evaluation stores the player it was evaluated for, plus the score.
#[derive(Clone, Debug)]
struct CfStateEval {
    score: i64,
    player: u8,
}

impl Evaluator<CfConfig> for CfEval {
    type StateEvaluation = CfStateEval;

    fn evaluate_new_state(
        &self,
        state: &ConnectFour,
        moves: &Vec<CfMove>,
        _: Option<SearchHandle<CfConfig>>,
    ) -> (Vec<()>, CfStateEval) {
        let player = state.current;
        let score = state.evaluate_for(player);
        (vec![(); moves.len()], CfStateEval { score, player })
    }

    fn interpret_evaluation_for_player(&self, evaln: &CfStateEval, player: &u8) -> i64 {
        if *player == evaln.player {
            evaln.score
        } else {
            -evaln.score
        }
    }

    fn evaluate_existing_state(
        &self,
        state: &ConnectFour,
        _evaln: &CfStateEval,
        _: SearchHandle<CfConfig>,
    ) -> CfStateEval {
        let player = state.current;
        CfStateEval {
            score: state.evaluate_for(player),
            player,
        }
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
}

// --- WASM API ---

#[wasm_bindgen]
pub struct ConnectFourWasm {
    manager: MCTSManager<CfConfig>,
    cols: usize,
    rows: usize,
    k: usize,
    num_players: usize,
}

impl Default for ConnectFourWasm {
    fn default() -> Self {
        Self::create(7, 6, 4, 2)
    }
}

#[wasm_bindgen]
impl ConnectFourWasm {
    fn create(cols: usize, rows: usize, k: usize, num_players: usize) -> Self {
        let cols = cols.clamp(3, MAX_COLS);
        let rows = rows.clamp(3, MAX_ROWS);
        let k = k.clamp(3, cols.max(rows));
        let num_players = num_players.clamp(2, MAX_PLAYERS);
        Self {
            manager: MCTSManager::new(
                ConnectFour::new(cols, rows, k, num_players),
                CfConfig,
                CfEval,
                UCTPolicy::new(1.4),
                (),
            ),
            cols,
            rows,
            k,
            num_players,
        }
    }

    #[wasm_bindgen(constructor)]
    pub fn new(cols: u32, rows: u32, k: u32, num_players: u32) -> Self {
        Self::create(cols as usize, rows as usize, k as usize, num_players as usize)
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
    pub fn num_players(&self) -> u32 {
        self.num_players as u32
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

    /// Board as string, top row first, left to right.
    /// ' '=empty, '1'=player 0, '2'=player 1, '3'=player 2, '4'=player 3.
    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    /// Current player as 0-indexed number string ("0", "1", etc.)
    pub fn current_player(&self) -> u32 {
        self.manager.tree().root_state().current as u32
    }

    pub fn is_terminal(&self) -> bool {
        let s = self.manager.tree().root_state();
        s.winner().is_some() || s.is_full()
    }

    /// Returns winner player number (0-indexed) as string, "Draw", or "" (not over).
    pub fn result(&self) -> String {
        let state = self.manager.tree().root_state();
        if let Some(winner) = state.winner() {
            format!("{}", winner + 1) // 1-indexed for display
        } else if state.is_full() {
            "Draw".into()
        } else {
            String::new()
        }
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    pub fn apply_move(&mut self, col: &str) -> bool {
        let col_num: u8 = match col.parse() {
            Ok(n) if (n as usize) < self.cols => n,
            _ => return false,
        };
        let m = CfMove(col_num);
        if self.manager.advance(&m).is_ok() {
            return true;
        }
        self.manager.playout_n(100);
        self.manager.advance(&m).is_ok()
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            ConnectFour::new(self.cols, self.rows, self.k, self.num_players),
            CfConfig,
            CfEval,
            UCTPolicy::new(1.4),
            (),
        );
    }
}
