use mcts::tree_policy::*;
use mcts::*;
use wasm_bindgen::prelude::*;

use crate::types;

// --- Shift: 3x3, each player has 3 pieces. Place phase then move phase. ---
// First 3 moves per player: place a piece on any empty cell.
// After that: pick one of your pieces and move it to any empty cell.
// Win: 3 in a row (like tic-tac-toe).

const BOARD_SIZE: usize = 9;
const PIECES_PER_PLAYER: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Empty,
    Player(u8),
}

const PLAYER_SYMBOLS: [char; 4] = ['X', 'O', 'A', 'B'];

const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // cols
    [0, 4, 8], [2, 4, 6],             // diags
];

/// Move: either Place(cell) or Shift(from, to).
/// Encoded as a single u16: high bit = is_shift, bits 6..3 = from, bits 2..0 = to (or cell).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ShiftMove {
    from: Option<u8>, // None = placement, Some(idx) = shift from
    to: u8,
}

impl std::fmt::Display for ShiftMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.from {
            None => write!(f, "P{}", self.to),       // Place at cell
            Some(fr) => write!(f, "M{},{}", fr, self.to), // Move from→to
        }
    }
}

impl ShiftMove {
    fn encode(&self) -> String {
        format!("{}", self)
    }

    fn decode(s: &str) -> Option<Self> {
        if let Some(rest) = s.strip_prefix('P') {
            let to: u8 = rest.parse().ok()?;
            if (to as usize) < BOARD_SIZE {
                return Some(ShiftMove { from: None, to });
            }
        } else if let Some(rest) = s.strip_prefix('M') {
            let parts: Vec<&str> = rest.split(',').collect();
            if parts.len() == 2 {
                let from: u8 = parts[0].parse().ok()?;
                let to: u8 = parts[1].parse().ok()?;
                if (from as usize) < BOARD_SIZE && (to as usize) < BOARD_SIZE {
                    return Some(ShiftMove {
                        from: Some(from),
                        to,
                    });
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug)]
struct ShiftGame {
    board: [Cell; BOARD_SIZE],
    current: u8,
    num_players: usize,
    pieces_placed: [u8; 4], // per player
}

impl ShiftGame {
    fn new(num_players: usize) -> Self {
        Self {
            board: [Cell::Empty; BOARD_SIZE],
            current: 0,
            num_players,
            pieces_placed: [0; 4],
        }
    }

    fn in_placement_phase(&self) -> bool {
        self.pieces_placed[self.current as usize] < PIECES_PER_PLAYER
    }

    fn winner(&self) -> Option<u8> {
        for line in &WIN_LINES {
            let a = self.board[line[0]];
            if a == Cell::Empty {
                continue;
            }
            if a == self.board[line[1]] && a == self.board[line[2]] {
                if let Cell::Player(p) = a {
                    return Some(p);
                }
            }
        }
        None
    }

    fn board_string(&self) -> String {
        self.board
            .iter()
            .map(|c| match c {
                Cell::Empty => ' ',
                Cell::Player(p) => PLAYER_SYMBOLS.get(*p as usize).copied().unwrap_or('?'),
            })
            .collect()
    }

    /// Evaluate from a specific player's perspective.
    fn evaluate_for(&self, player: u8) -> i64 {
        let my_cell = Cell::Player(player);
        let mut score: i64 = 0;

        for line in &WIN_LINES {
            let mut mine = 0;
            let mut empty = 0;
            let mut theirs = 0;
            for &idx in line {
                match self.board[idx] {
                    c if c == my_cell => mine += 1,
                    Cell::Empty => empty += 1,
                    _ => theirs += 1,
                }
            }
            if mine == 3 {
                score += 1000;
            } else if theirs == 3 {
                score -= 1000;
            } else if mine == 2 && empty == 1 {
                score += 50;
            } else if theirs == 2 && empty == 1 {
                score -= 80;
            } else if mine == 1 && empty == 2 {
                score += 5;
            }
        }

        // Center bonus
        if self.board[4] == my_cell {
            score += 10;
        }

        score
    }
}

impl GameState for ShiftGame {
    type Move = ShiftMove;
    type Player = u8;
    type MoveList = Vec<ShiftMove>;

    fn current_player(&self) -> u8 {
        self.current
    }

    fn available_moves(&self) -> Vec<ShiftMove> {
        if self.winner().is_some() {
            return vec![];
        }

        let mut moves = Vec::new();

        if self.in_placement_phase() {
            // Place on any empty cell
            for i in 0..BOARD_SIZE {
                if self.board[i] == Cell::Empty {
                    moves.push(ShiftMove {
                        from: None,
                        to: i as u8,
                    });
                }
            }
        } else {
            // Move one of my pieces to any empty cell
            let my_cell = Cell::Player(self.current);
            let my_pieces: Vec<u8> = (0..BOARD_SIZE as u8)
                .filter(|&i| self.board[i as usize] == my_cell)
                .collect();
            let empty_cells: Vec<u8> = (0..BOARD_SIZE as u8)
                .filter(|&i| self.board[i as usize] == Cell::Empty)
                .collect();

            for &from in &my_pieces {
                for &to in &empty_cells {
                    moves.push(ShiftMove {
                        from: Some(from),
                        to,
                    });
                }
            }
        }

        moves
    }

    fn make_move(&mut self, mov: &ShiftMove) {
        match mov.from {
            None => {
                // Placement
                self.board[mov.to as usize] = Cell::Player(self.current);
                self.pieces_placed[self.current as usize] += 1;
            }
            Some(from) => {
                // Shift
                self.board[from as usize] = Cell::Empty;
                self.board[mov.to as usize] = Cell::Player(self.current);
            }
        }
        self.current = (self.current + 1) % self.num_players as u8;
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if self.winner().is_some() {
            Some(ProvenValue::Loss) // winner just moved, current player lost
        } else {
            None
        }
    }
}

// --- Evaluator ---

struct ShiftEval;

#[derive(Clone, Debug)]
struct ShiftStateEval {
    score: i64,
    player: u8,
}

impl Evaluator<ShiftConfig> for ShiftEval {
    type StateEvaluation = ShiftStateEval;

    fn evaluate_new_state(
        &self,
        state: &ShiftGame,
        moves: &Vec<ShiftMove>,
        _: Option<SearchHandle<ShiftConfig>>,
    ) -> (Vec<()>, ShiftStateEval) {
        let player = state.current;
        (
            vec![(); moves.len()],
            ShiftStateEval {
                score: state.evaluate_for(player),
                player,
            },
        )
    }

    fn interpret_evaluation_for_player(&self, evaln: &ShiftStateEval, player: &u8) -> i64 {
        if *player == evaln.player {
            evaln.score
        } else {
            -evaln.score
        }
    }

    fn evaluate_existing_state(
        &self,
        state: &ShiftGame,
        _evaln: &ShiftStateEval,
        _: SearchHandle<ShiftConfig>,
    ) -> ShiftStateEval {
        let player = state.current;
        ShiftStateEval {
            score: state.evaluate_for(player),
            player,
        }
    }
}

// --- MCTS Config ---

#[derive(Default)]
struct ShiftConfig;

impl MCTS for ShiftConfig {
    type State = ShiftGame;
    type Eval = ShiftEval;
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
pub struct ShiftWasm {
    manager: MCTSManager<ShiftConfig>,
    num_players: usize,
}

impl Default for ShiftWasm {
    fn default() -> Self {
        Self::create(2)
    }
}

#[wasm_bindgen]
impl ShiftWasm {
    fn create(num_players: usize) -> Self {
        let num_players = num_players.clamp(2, 4);
        Self {
            manager: MCTSManager::new(
                ShiftGame::new(num_players),
                ShiftConfig,
                ShiftEval,
                UCTPolicy::new(1.4),
                (),
            ),
            num_players,
        }
    }

    #[wasm_bindgen(constructor)]
    pub fn new(num_players: u32) -> Self {
        Self::create(num_players as usize)
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
        let tree = types::export_tree::<ShiftConfig>(
            self.manager.tree().root_node(),
            max_depth,
            &|_| None,
        );
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    /// Board as 9-char string: ' '=empty, 'X'=p0, 'O'=p1, 'A'=p2, 'B'=p3.
    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    pub fn current_player(&self) -> u32 {
        self.manager.tree().root_state().current as u32
    }

    /// Is the current player still placing pieces?
    pub fn in_placement_phase(&self) -> bool {
        self.manager.tree().root_state().in_placement_phase()
    }

    pub fn is_terminal(&self) -> bool {
        self.manager.tree().root_state().winner().is_some()
    }

    /// Returns winner as "1","2",etc. or "" (not over).
    pub fn result(&self) -> String {
        if let Some(w) = self.manager.tree().root_state().winner() {
            format!("{}", w + 1)
        } else {
            String::new()
        }
    }

    pub fn root_proven_value(&self) -> String {
        format!("{:?}", self.manager.root_proven_value())
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| m.encode())
    }

    /// Apply a move. Format: "P4" (place at cell 4) or "M0,4" (move from 0 to 4).
    /// Apply a move. Creates a fresh search tree each time to prevent
    /// unbounded tree growth (Shift games can cycle indefinitely).
    pub fn apply_move(&mut self, mov: &str) -> bool {
        if let Some(m) = ShiftMove::decode(mov) {
            let mut state = self.manager.tree().root_state().clone();
            let legal = state.available_moves();
            if !legal.contains(&m) {
                return false;
            }
            state.make_move(&m);
            self.manager = MCTSManager::new(
                state,
                ShiftConfig,
                ShiftEval,
                UCTPolicy::new(1.4),
                (),
            );
            true
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            ShiftGame::new(self.num_players),
            ShiftConfig,
            ShiftEval,
            UCTPolicy::new(1.4),
            (),
        );
    }
}
