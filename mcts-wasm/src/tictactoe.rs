use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Generalized M,N,K game: M cols x N rows, K in a row, P players ---

const MAX_COLS: usize = 10;
const MAX_ROWS: usize = 10;
const MAX_CELLS: usize = MAX_COLS * MAX_ROWS;
const MAX_PLAYERS: usize = 4;

const PLAYER_SYMBOLS: [char; MAX_PLAYERS] = ['X', 'O', 'A', 'B'];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Empty,
    Player(u8),
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
    current: u8,
    cols: usize,
    rows: usize,
    k: usize,
    num_players: usize,
}

impl TicTacToe {
    fn new(cols: usize, rows: usize, k: usize, num_players: usize) -> Self {
        Self {
            board: [Cell::Empty; MAX_CELLS],
            current: 0,
            cols,
            rows,
            k,
            num_players,
        }
    }

    fn cell_count(&self) -> usize {
        self.cols * self.rows
    }

    fn winner(&self) -> Option<u8> {
        let dirs: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for r in 0..self.rows {
            for c in 0..self.cols {
                if let Cell::Player(p) = self.board[r * self.cols + c] {
                    for &(dr, dc) in &dirs {
                        let mut count = 1usize;
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
                            if self.board[nr as usize * self.cols + nc as usize] == Cell::Player(p)
                            {
                                count += 1;
                            } else {
                                break;
                            }
                        }
                        if count >= self.k {
                            return Some(p);
                        }
                    }
                }
            }
        }
        None
    }

    fn board_full(&self) -> bool {
        (0..self.cell_count()).all(|i| self.board[i] != Cell::Empty)
    }

    fn result_str(&self) -> String {
        if let Some(w) = self.winner() {
            format!("{}", w + 1) // 1-indexed for display
        } else if self.board_full() {
            "Draw".to_string()
        } else {
            String::new()
        }
    }

    fn board_string(&self) -> String {
        (0..self.cell_count())
            .map(|i| match self.board[i] {
                Cell::Empty => ' ',
                Cell::Player(p) => PLAYER_SYMBOLS.get(p as usize).copied().unwrap_or('?'),
            })
            .collect()
    }

    /// Evaluate from a specific player's perspective.
    fn evaluate_for(&self, player: u8) -> i64 {
        let my_cell = Cell::Player(player);
        let mut score: i64 = 0;

        let dirs: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];
        for r in 0..self.rows {
            for c in 0..self.cols {
                for &(dr, dc) in &dirs {
                    // Check if window fits
                    let end_r = r as i32 + dr * (self.k as i32 - 1);
                    let end_c = c as i32 + dc * (self.k as i32 - 1);
                    if end_r < 0
                        || end_r >= self.rows as i32
                        || end_c < 0
                        || end_c >= self.cols as i32
                    {
                        continue;
                    }

                    let mut mine = 0usize;
                    let mut empty = 0usize;
                    let mut theirs = 0usize;
                    for step in 0..self.k {
                        let idx = (r as i32 + dr * step as i32) as usize * self.cols
                            + (c as i32 + dc * step as i32) as usize;
                        match self.board[idx] {
                            c if c == my_cell => mine += 1,
                            Cell::Empty => empty += 1,
                            _ => theirs += 1,
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
                    } else if mine >= 2 && theirs == 0 {
                        score += 5;
                    }
                }
            }
        }

        score
    }
}

impl GameState for TicTacToe {
    type Move = TttMove;
    type Player = u8;
    type MoveList = Vec<TttMove>;

    fn current_player(&self) -> u8 {
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
        self.board[mov.0 as usize] = Cell::Player(self.current);
        self.current = (self.current + 1) % self.num_players as u8;
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

#[derive(Clone, Debug)]
struct TttStateEval {
    score: i64,
    player: u8,
}

impl Evaluator<TttConfig> for TttEval {
    type StateEvaluation = TttStateEval;

    fn evaluate_new_state(
        &self,
        state: &TicTacToe,
        moves: &Vec<TttMove>,
        _: Option<SearchHandle<TttConfig>>,
    ) -> (Vec<()>, TttStateEval) {
        let player = state.current;
        (
            vec![(); moves.len()],
            TttStateEval {
                score: state.evaluate_for(player),
                player,
            },
        )
    }

    fn interpret_evaluation_for_player(&self, evaln: &TttStateEval, player: &u8) -> i64 {
        if *player == evaln.player {
            evaln.score
        } else {
            -evaln.score
        }
    }

    fn evaluate_existing_state(
        &self,
        state: &TicTacToe,
        _evaln: &TttStateEval,
        _: SearchHandle<TttConfig>,
    ) -> TttStateEval {
        let player = state.current;
        TttStateEval {
            score: state.evaluate_for(player),
            player,
        }
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
    num_players: usize,
}

impl Default for TicTacToeWasm {
    fn default() -> Self {
        Self::create(3, 3, 3, 2)
    }
}

#[wasm_bindgen]
impl TicTacToeWasm {
    fn create(cols: usize, rows: usize, k: usize, num_players: usize) -> Self {
        let cols = cols.clamp(2, MAX_COLS);
        let rows = rows.clamp(2, MAX_ROWS);
        let k = k.clamp(2, cols.max(rows));
        let num_players = num_players.clamp(2, MAX_PLAYERS);
        Self {
            manager: MCTSManager::new(
                TicTacToe::new(cols, rows, k, num_players),
                TttConfig,
                TttEval,
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
            types::export_tree::<TttConfig>(self.manager.tree().root_node(), max_depth, &|_| None);
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    /// Board as string: ' '=empty, 'X'=p0, 'O'=p1, 'A'=p2, 'B'=p3. Row-major.
    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    pub fn current_player(&self) -> u32 {
        self.manager.tree().root_state().current as u32
    }

    pub fn is_terminal(&self) -> bool {
        let state = self.manager.tree().root_state();
        state.winner().is_some() || state.board_full()
    }

    /// Returns winner as "1","2",etc., "Draw", or "" (not over).
    pub fn result(&self) -> String {
        self.manager.tree().root_state().result_str()
    }

    pub fn root_proven_value(&self) -> String {
        format!("{:?}", self.manager.root_proven_value())
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

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
            TicTacToe::new(self.cols, self.rows, self.k, self.num_players),
            TttConfig,
            TttEval,
            UCTPolicy::new(1.4),
            (),
        );
    }
}
