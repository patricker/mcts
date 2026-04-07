use mcts::tree_policy::*;
use mcts::*;
use rand::Rng;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Directions ---

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
}

impl std::fmt::Display for Dir {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Dir::Up => write!(f, "Up"),
            Dir::Down => write!(f, "Down"),
            Dir::Left => write!(f, "Left"),
            Dir::Right => write!(f, "Right"),
        }
    }
}

const ALL_DIRS: [Dir; 4] = [Dir::Up, Dir::Down, Dir::Left, Dir::Right];

// --- Game State ---

#[derive(Clone, Debug)]
struct Game2048 {
    board: [[u32; 4]; 4],
    score: u32,
}

impl Game2048 {
    fn new() -> Self {
        let mut g = Game2048 {
            board: [[0; 4]; 4],
            score: 0,
        };
        g.spawn_tile();
        g.spawn_tile();
        g
    }

    fn empty_cells(&self) -> Vec<(usize, usize)> {
        let mut cells = Vec::new();
        for r in 0..4 {
            for c in 0..4 {
                if self.board[r][c] == 0 {
                    cells.push((r, c));
                }
            }
        }
        cells
    }

    fn spawn_tile(&mut self) {
        let empty = self.empty_cells();
        if empty.is_empty() {
            return;
        }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..empty.len());
        let (r, c) = empty[idx];
        self.board[r][c] = if rng.gen::<f64>() < 0.9 { 2 } else { 4 };
    }

    fn max_tile(&self) -> u32 {
        self.board.iter().flat_map(|row| row.iter()).copied().max().unwrap_or(0)
    }

    /// Slide a single row left, returning (new_row, points_scored).
    fn slide_row_left(row: &[u32; 4]) -> ([u32; 4], u32) {
        // Step 1: compact (remove zeros)
        let mut compacted = [0u32; 4];
        let mut pos = 0;
        for &val in row {
            if val != 0 {
                compacted[pos] = val;
                pos += 1;
            }
        }

        // Step 2: merge adjacent equal tiles left to right
        let mut score = 0u32;
        let mut merged = [0u32; 4];
        let mut out_pos = 0;
        let mut i = 0;
        while i < 4 {
            if compacted[i] == 0 {
                break;
            }
            if i + 1 < 4 && compacted[i] == compacted[i + 1] {
                let val = compacted[i] * 2;
                merged[out_pos] = val;
                score += val;
                i += 2;
            } else {
                merged[out_pos] = compacted[i];
                i += 1;
            }
            out_pos += 1;
        }

        (merged, score)
    }

    /// Apply a direction to the board. Returns true if the board changed.
    fn slide(&mut self, dir: Dir) -> bool {
        let old_board = self.board;

        match dir {
            Dir::Left => {
                for r in 0..4 {
                    let (new_row, pts) = Self::slide_row_left(&self.board[r]);
                    self.board[r] = new_row;
                    self.score += pts;
                }
            }
            Dir::Right => {
                for r in 0..4 {
                    let mut row = self.board[r];
                    row.reverse();
                    let (mut new_row, pts) = Self::slide_row_left(&row);
                    new_row.reverse();
                    self.board[r] = new_row;
                    self.score += pts;
                }
            }
            Dir::Up => {
                for c in 0..4 {
                    let col = [self.board[0][c], self.board[1][c], self.board[2][c], self.board[3][c]];
                    let (new_col, pts) = Self::slide_row_left(&col);
                    for (r, &val) in new_col.iter().enumerate() {
                        self.board[r][c] = val;
                    }
                    self.score += pts;
                }
            }
            Dir::Down => {
                for c in 0..4 {
                    let mut col = [self.board[0][c], self.board[1][c], self.board[2][c], self.board[3][c]];
                    col.reverse();
                    let (mut new_col, pts) = Self::slide_row_left(&col);
                    new_col.reverse();
                    for (r, &val) in new_col.iter().enumerate() {
                        self.board[r][c] = val;
                    }
                    self.score += pts;
                }
            }
        }

        self.board != old_board
    }

    /// Check if a direction would change the board (without mutating).
    fn would_change(&self, dir: Dir) -> bool {
        let mut clone = self.clone();
        clone.slide(dir)
    }

    fn has_adjacent_equal(&self) -> bool {
        for r in 0..4 {
            for c in 0..4 {
                let val = self.board[r][c];
                if c + 1 < 4 && self.board[r][c + 1] == val {
                    return true;
                }
                if r + 1 < 4 && self.board[r + 1][c] == val {
                    return true;
                }
            }
        }
        false
    }

    fn is_game_over(&self) -> bool {
        self.empty_cells().is_empty() && !self.has_adjacent_equal()
    }
}

// --- MCTS integration ---

impl GameState for Game2048 {
    type Move = Dir;
    type Player = ();
    type MoveList = Vec<Dir>;

    fn current_player(&self) {}

    fn available_moves(&self) -> Vec<Dir> {
        if self.is_game_over() {
            return vec![];
        }
        ALL_DIRS
            .iter()
            .copied()
            .filter(|&d| self.would_change(d))
            .collect()
    }

    fn make_move(&mut self, mov: &Dir) {
        let changed = self.slide(*mov);
        if changed {
            self.spawn_tile();
        }
    }
}

struct Game2048Eval;

impl Evaluator<Game2048Config> for Game2048Eval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &Game2048,
        moves: &Vec<Dir>,
        _: Option<SearchHandle<Game2048Config>>,
    ) -> (Vec<()>, i64) {
        let value = state.score as i64
            + state.empty_cells().len() as i64 * 10
            + state.max_tile() as i64;
        (vec![(); moves.len()], value)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self,
        state: &Game2048,
        _: &i64,
        _: SearchHandle<Game2048Config>,
    ) -> i64 {
        state.score as i64 + state.empty_cells().len() as i64 * 10 + state.max_tile() as i64
    }
}

#[derive(Default)]
struct Game2048Config;

impl MCTS for Game2048Config {
    type State = Game2048;
    type Eval = Game2048Eval;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();

    fn solver_enabled(&self) -> bool {
        false
    }

    fn max_playout_depth(&self) -> usize {
        50
    }
}

// --- WASM API ---

#[wasm_bindgen]
pub struct Game2048Wasm {
    manager: MCTSManager<Game2048Config>,
}

impl Default for Game2048Wasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Game2048Wasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            manager: MCTSManager::new(
                Game2048::new(),
                Game2048Config,
                Game2048Eval,
                UCTPolicy::new(1.0),
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

    /// Returns the board as a flat array of 16 u32 values (row-major, top to bottom).
    pub fn get_board(&self) -> JsValue {
        let state = self.manager.tree().root_state();
        let flat: Vec<u32> = state.board.iter().flat_map(|row| row.iter()).copied().collect();
        serde_wasm_bindgen::to_value(&flat).unwrap()
    }

    pub fn score(&self) -> u32 {
        self.manager.tree().root_state().score
    }

    pub fn max_tile(&self) -> u32 {
        self.manager.tree().root_state().max_tile()
    }

    pub fn is_terminal(&self) -> bool {
        self.manager.tree().root_state().is_game_over()
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    /// Apply a move by mutating the state and creating a fresh search tree.
    /// 2048 is stochastic (random tile spawns), so tree reuse via advance()
    /// doesn't work — each playout sees different random outcomes. Instead,
    /// we apply the move to get the actual next state and start fresh.
    pub fn apply_move(&mut self, dir: &str) -> bool {
        let d = match dir {
            "Up" => Dir::Up,
            "Down" => Dir::Down,
            "Left" => Dir::Left,
            "Right" => Dir::Right,
            _ => return false,
        };
        let mut state = self.manager.tree().root_state().clone();
        if !state.would_change(d) {
            return false;
        }
        state.slide(d);
        state.spawn_tile();
        self.manager = MCTSManager::new(
            state,
            Game2048Config,
            Game2048Eval,
            UCTPolicy::new(1.0),
            (),
        );
        true
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            Game2048::new(),
            Game2048Config,
            Game2048Eval,
            UCTPolicy::new(1.0),
            (),
        );
    }
}
