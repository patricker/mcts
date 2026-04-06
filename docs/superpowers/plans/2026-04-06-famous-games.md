# Famous Games for MCTS Playground

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Tic-Tac-Toe, Connect Four, and 2048 to the WASM playground as recognizable, playable games that showcase MCTS capabilities (solver, deep search, chance nodes).

**Architecture:** Each game is a self-contained Rust module in `mcts-wasm/src/` implementing `GameState` + evaluator + MCTS config + `#[wasm_bindgen]` wrapper, plus a React demo component in `docs/src/components/demos/` and a tab in the playground. Follow existing patterns from `nim.rs`/`NimSolverDemo.tsx`.

**Tech Stack:** Rust (mcts core), wasm-bindgen, React/TypeScript, Docusaurus, CSS Modules

---

## File Structure

### Tic-Tac-Toe
| File | Responsibility |
|------|---------------|
| `mcts-wasm/src/tictactoe.rs` | Game logic, evaluator, MCTS config, WASM bindings |
| `docs/src/components/demos/TicTacToeDemo.tsx` | Interactive demo: human vs MCTS |
| `docs/src/components/demos/TicTacToeDemo.module.css` | Board grid styling |

### Connect Four
| File | Responsibility |
|------|---------------|
| `mcts-wasm/src/connectfour.rs` | Game logic, evaluator, MCTS config, WASM bindings |
| `docs/src/components/demos/ConnectFourDemo.tsx` | Interactive demo: human vs MCTS |
| `docs/src/components/demos/ConnectFourDemo.module.css` | Board grid styling |

### 2048
| File | Responsibility |
|------|---------------|
| `mcts-wasm/src/game2048.rs` | Game logic with chance nodes, evaluator, MCTS config, WASM bindings |
| `docs/src/components/demos/Game2048Demo.tsx` | Interactive demo: MCTS suggests moves |
| `docs/src/components/demos/Game2048Demo.module.css` | Tile grid styling |

### Shared modifications
| File | Change |
|------|--------|
| `mcts-wasm/src/lib.rs` | Add `mod` and `pub use` for 3 new modules |
| `docs/src/pages/playground.tsx` | Add 3 tabs + DemoLoader cases |

---

## Game Design Decisions

**Tic-Tac-Toe:**
- Human plays X (first), MCTS plays O
- Solver enabled — MCTS proves draws/wins
- 3x3 board as 9-char string ("XO X O   ")
- Moves: "0"-"8" (cell index)

**Connect Four:**
- Human plays Red (first), MCTS plays Yellow
- Solver disabled (tree too deep for proof in browser)
- 7x6 board as 42-char string, column-major or row-major
- Moves: "0"-"6" (column index)
- Gravity: pieces drop to lowest empty row

**2048:**
- Single-player, MCTS suggests best direction
- Chance nodes for tile spawning (90% "2", 10% "4")
- Open-loop (not closed-loop) — simpler, works well for 2048
- 4x4 board as JSON array
- Moves: "Up", "Down", "Left", "Right"
- Human makes all moves, MCTS analyzes before each

---

### Task 1: Tic-Tac-Toe Rust + WASM

**Files:**
- Create: `mcts-wasm/src/tictactoe.rs`
- Modify: `mcts-wasm/src/lib.rs`

- [ ] **Step 1: Create the game module**

Create `mcts-wasm/src/tictactoe.rs` with the complete game implementation:

```rust
use mcts::*;
use mcts::tree_policy::UCTPolicy;
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;

mod types;
use crate::types;

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------

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

#[derive(Clone, Debug)]
struct TicTacToe {
    board: [Cell; 9],
    current: Player,
}

#[derive(Clone, Debug, PartialEq)]
struct TttMove(u8); // 0-8

impl std::fmt::Display for TttMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TicTacToe {
    fn new() -> Self {
        Self {
            board: [Cell::Empty; 9],
            current: Player::X,
        }
    }

    fn winner(&self) -> Option<Player> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8], // cols
            [0, 4, 8], [2, 4, 6],             // diags
        ];
        for line in &LINES {
            let a = self.board[line[0]];
            if a != Cell::Empty && a == self.board[line[1]] && a == self.board[line[2]] {
                return match a {
                    Cell::X => Some(Player::X),
                    Cell::O => Some(Player::O),
                    Cell::Empty => unreachable!(),
                };
            }
        }
        None
    }

    fn board_full(&self) -> bool {
        self.board.iter().all(|&c| c != Cell::Empty)
    }

    fn board_string(&self) -> String {
        self.board.iter().map(|c| match c {
            Cell::Empty => ' ',
            Cell::X => 'X',
            Cell::O => 'O',
        }).collect()
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
        self.board.iter().enumerate()
            .filter(|(_, &c)| c == Cell::Empty)
            .map(|(i, _)| TttMove(i as u8))
            .collect()
    }

    fn make_move(&mut self, mov: &TttMove) {
        self.board[mov.0 as usize] = match self.current {
            Player::X => Cell::X,
            Player::O => Cell::O,
        };
        self.current = match self.current {
            Player::X => Player::O,
            Player::O => Player::X,
        };
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if let Some(winner) = self.winner() {
            // The winner just moved, so current_player is the loser
            return Some(ProvenValue::Loss);
        }
        if self.board_full() {
            return Some(ProvenValue::Draw);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Evaluator + Config
// ---------------------------------------------------------------------------

struct TttEval;

impl Evaluator<TttConfig> for TttEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        _state: &TicTacToe,
        moves: &Vec<TttMove>,
        _: Option<SearchHandle<TttConfig>>,
    ) -> (Vec<()>, i64) {
        (vec![(); moves.len()], 0)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &Player) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self, _: &TicTacToe, evaln: &i64, _: SearchHandle<TttConfig>,
    ) -> i64 {
        *evaln
    }
}

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

// ---------------------------------------------------------------------------
// WASM bindings
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct TicTacToeWasm {
    manager: MCTSManager<TttConfig>,
}

#[wasm_bindgen]
impl TicTacToeWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            manager: MCTSManager::new(
                TicTacToe::new(),
                TttConfig,
                TttEval,
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

    pub fn get_tree(&self, max_depth: u32) -> JsValue {
        let tree = types::export_tree::<TttConfig>(
            self.manager.tree().root_node(), max_depth, &|_| None,
        );
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    pub fn current_player(&self) -> String {
        format!("{:?}", self.manager.tree().root_state().current)
    }

    pub fn is_terminal(&self) -> bool {
        let state = self.manager.tree().root_state();
        state.winner().is_some() || state.board_full()
    }

    /// Returns "X", "O", "Draw", or "" (game not over)
    pub fn result(&self) -> String {
        let state = self.manager.tree().root_state();
        if let Some(winner) = state.winner() {
            format!("{:?}", winner)
        } else if state.board_full() {
            "Draw".to_string()
        } else {
            String::new()
        }
    }

    pub fn root_proven_value(&self) -> String {
        format!("{:?}", self.manager.root_proven_value())
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    pub fn apply_move(&mut self, mov: &str) -> bool {
        if let Ok(idx) = mov.parse::<u8>() {
            if idx < 9 {
                self.manager.advance(&TttMove(idx)).is_ok()
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            TicTacToe::new(), TttConfig, TttEval, UCTPolicy::new(1.0), (),
        );
    }
}
```

- [ ] **Step 2: Register the module in lib.rs**

Add to `mcts-wasm/src/lib.rs`:
- `mod tictactoe;` in the module declarations
- `pub use tictactoe::TicTacToeWasm;` in the re-exports

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p mcts-wasm`
Expected: Compiles with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add mcts-wasm/src/tictactoe.rs mcts-wasm/src/lib.rs
git commit -m "feat: Tic-Tac-Toe WASM game with solver"
```

---

### Task 2: Tic-Tac-Toe React Demo

**Files:**
- Create: `docs/src/components/demos/TicTacToeDemo.tsx`
- Create: `docs/src/components/demos/TicTacToeDemo.module.css`
- Modify: `docs/src/pages/playground.tsx`

- [ ] **Step 1: Create the CSS module**

Create `docs/src/components/demos/TicTacToeDemo.module.css`:

```css
.container {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
}

.boardSection {
  flex: 0 0 auto;
}

.statsSection {
  flex: 1;
  min-width: 200px;
}

.status {
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}

.grid {
  display: grid;
  grid-template-columns: repeat(3, 64px);
  grid-template-rows: repeat(3, 64px);
  gap: 4px;
}

.cell {
  width: 64px;
  height: 64px;
  font-size: 1.75rem;
  font-weight: 700;
  border: 2px solid var(--ifm-color-emphasis-300);
  border-radius: 4px;
  cursor: pointer;
  background: var(--ifm-background-surface-color);
  color: var(--ifm-font-color-base);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.1s;
}

.cell:hover:not(:disabled) {
  background: var(--ifm-color-emphasis-100);
}

.cell:disabled {
  cursor: default;
}

.cellX {
  color: #3b82f6;
}

.cellO {
  color: #ef4444;
}

.controls {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}

.btn {
  padding: 0.375rem 0.75rem;
  font-size: 0.8rem;
  border: 1px solid var(--ifm-color-emphasis-300);
  border-radius: 4px;
  background: var(--ifm-background-surface-color);
  color: var(--ifm-font-color-base);
  cursor: pointer;
}

.btn:hover {
  background: var(--ifm-color-emphasis-100);
}

.provenBadge {
  font-family: var(--ifm-font-family-monospace);
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  display: inline-block;
  margin-bottom: 0.5rem;
}

.provenWin {
  color: #22c55e;
  background: rgba(34, 197, 94, 0.1);
}

.provenLoss {
  color: #ef4444;
  background: rgba(239, 68, 68, 0.1);
}

.provenDraw {
  color: #eab308;
  background: rgba(234, 179, 8, 0.1);
}

.moveList {
  font-size: 0.8rem;
  margin-top: 0.5rem;
}

.moveRow {
  display: flex;
  justify-content: space-between;
  padding: 0.2rem 0;
  border-bottom: 1px solid var(--ifm-color-emphasis-100);
}

.gameOver {
  margin-top: 0.75rem;
  padding: 0.75rem;
  text-align: center;
  border-radius: 4px;
  background: var(--ifm-color-emphasis-100);
  font-weight: 600;
}
```

- [ ] **Step 2: Create the demo component**

Create `docs/src/components/demos/TicTacToeDemo.tsx`:

```tsx
import { useState, useEffect, useRef, useCallback } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './TicTacToeDemo.module.css';

interface ChildStat {
  mov: string;
  visits: number;
  avg_reward: number;
  proven?: string;
}

interface Stats {
  total_playouts: number;
  total_nodes: number;
  best_move?: string;
  children: ChildStat[];
}

const PLAYOUTS = 5000;

function TicTacToeDemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');
  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);
  const [board, setBoard] = useState(' '.repeat(9));
  const [currentPlayer, setCurrentPlayer] = useState('X');
  const [stats, setStats] = useState<Stats | null>(null);
  const [proven, setProven] = useState('');
  const [gameResult, setGameResult] = useState('');
  const [thinking, setThinking] = useState(false);

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
    setCurrentPlayer(gameRef.current.current_player());
    setGameResult(gameRef.current.result());
    gameRef.current.playout_n(PLAYOUTS);
    setStats(gameRef.current.get_stats());
    setProven(gameRef.current.root_proven_value());
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) gameRef.current.free();
    gameRef.current = new wasm.TicTacToeWasm();
    setThinking(false);
    syncState();
  }, [wasm, syncState]);

  useEffect(() => {
    if (ready) initGame();
    return () => {
      if (gameRef.current) {
        gameRef.current.free();
        gameRef.current = null;
      }
    };
  }, [ready, initGame]);

  const handleCellClick = useCallback((idx: number) => {
    if (!gameRef.current || thinking || gameResult) return;
    const b = gameRef.current.get_board();
    if (b[idx] !== ' ') return;

    // Human move
    gameRef.current.apply_move(idx.toString());
    syncState();

    if (gameRef.current.result()) return;

    // MCTS response
    setThinking(true);
    setTimeout(() => {
      if (!gameRef.current) return;
      const best = gameRef.current.best_move();
      if (best) {
        gameRef.current.apply_move(best);
        syncState();
      }
      setThinking(false);
    }, 100);
  }, [thinking, gameResult, syncState]);

  if (error) return <div>Failed to load WASM: {error}</div>;
  if (!ready) return <div className={styles.status}>Loading WASM...</div>;

  const provenLabel = proven === 'Win' ? 'MCTS wins' :
                      proven === 'Loss' ? 'You win' :
                      proven === 'Draw' ? 'Proven draw' : '';

  return (
    <div className={styles.container}>
      <div className={styles.boardSection}>
        <div className={styles.status}>
          {gameResult
            ? null
            : thinking
              ? 'MCTS is thinking...'
              : `Your turn (${currentPlayer === 'X' ? 'X' : 'waiting...'})`}
        </div>

        <div className={styles.grid}>
          {Array.from(board).map((cell, i) => (
            <button
              key={i}
              className={`${styles.cell} ${cell === 'X' ? styles.cellX : cell === 'O' ? styles.cellO : ''}`}
              onClick={() => handleCellClick(i)}
              disabled={thinking || !!gameResult || cell !== ' ' || currentPlayer !== 'X'}
            >
              {cell !== ' ' ? cell : ''}
            </button>
          ))}
        </div>

        {gameResult && (
          <div className={styles.gameOver}>
            {gameResult === 'Draw' ? "It's a draw!" :
             gameResult === 'X' ? 'X wins!' : 'O wins!'}
          </div>
        )}

        <div className={styles.controls}>
          <button className={styles.btn} onClick={initGame}>New Game</button>
        </div>
      </div>

      <div className={styles.statsSection}>
        {provenLabel && (
          <div className={`${styles.provenBadge} ${
            proven === 'Win' ? styles.provenLoss :
            proven === 'Loss' ? styles.provenWin :
            styles.provenDraw
          }`}>
            {provenLabel}
          </div>
        )}

        {stats && stats.children.length > 0 && (
          <div className={styles.moveList}>
            <div style={{ fontWeight: 600, marginBottom: '0.25rem', fontSize: '0.75rem' }}>
              MCTS Analysis ({stats.total_playouts.toLocaleString()} playouts, {stats.total_nodes.toLocaleString()} nodes)
            </div>
            {stats.children
              .filter(c => c.visits > 0)
              .sort((a, b) => b.visits - a.visits)
              .map(c => (
                <div key={c.mov} className={styles.moveRow}>
                  <span>Cell {c.mov}</span>
                  <span>{c.visits} visits ({(c.avg_reward).toFixed(1)})</span>
                  {c.proven && <span>{c.proven}</span>}
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function TicTacToeDemo() {
  return (
    <BrowserOnly fallback={<div className={styles.status}>Loading...</div>}>
      {() => {
        const { WasmProvider } = require('../mcts/WasmProvider');
        return (
          <WasmProvider>
            <TicTacToeDemoInner />
          </WasmProvider>
        );
      }}
    </BrowserOnly>
  );
}
```

- [ ] **Step 3: Add Tic-Tac-Toe tab to playground**

In `docs/src/pages/playground.tsx`:

Add to tabs array:
```typescript
{ id: 'tictactoe', label: 'Tic-Tac-Toe' },
```

Add case to DemoLoader:
```typescript
case 'tictactoe': {
  const TicTacToeDemo =
    require('@site/src/components/demos/TicTacToeDemo').default;
  return <TicTacToeDemo />;
}
```

- [ ] **Step 4: Verify docs site builds**

Run: `cd docs && npm run build`
Expected: Build succeeds (WASM may not be available during build, but TSX should compile).

- [ ] **Step 5: Commit**

```bash
git add docs/src/components/demos/TicTacToeDemo.tsx docs/src/components/demos/TicTacToeDemo.module.css docs/src/pages/playground.tsx
git commit -m "feat: Tic-Tac-Toe playground demo with solver analysis"
```

---

### Task 3: Connect Four Rust + WASM

**Files:**
- Create: `mcts-wasm/src/connectfour.rs`
- Modify: `mcts-wasm/src/lib.rs`

- [ ] **Step 1: Create the game module**

Create `mcts-wasm/src/connectfour.rs`:

```rust
use mcts::*;
use mcts::tree_policy::UCTPolicy;
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;

use crate::types;

// ---------------------------------------------------------------------------
// Game state — 7 columns x 6 rows, gravity drop, 4-in-a-row wins
// ---------------------------------------------------------------------------

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

#[derive(Clone, Debug)]
struct ConnectFour {
    // board[row][col], row 0 = bottom
    board: [[Cell; 7]; 6],
    current: Player,
}

#[derive(Clone, Debug, PartialEq)]
struct CfMove(u8); // column 0-6

impl std::fmt::Display for CfMove {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl ConnectFour {
    fn new() -> Self {
        Self {
            board: [[Cell::Empty; 7]; 6],
            current: Player::Red,
        }
    }

    /// Find the lowest empty row in a column, or None if full.
    fn drop_row(&self, col: usize) -> Option<usize> {
        (0..6).find(|&row| self.board[row][col] == Cell::Empty)
    }

    /// Check if placing at (row, col) creates 4-in-a-row.
    fn check_win(&self, row: usize, col: usize) -> bool {
        let cell = self.board[row][col];
        if cell == Cell::Empty {
            return false;
        }

        const DIRS: [(i32, i32); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
        for (dr, dc) in DIRS {
            let mut count = 1;
            // Forward
            for step in 1..4 {
                let r = row as i32 + dr * step;
                let c = col as i32 + dc * step;
                if r < 0 || r >= 6 || c < 0 || c >= 7 {
                    break;
                }
                if self.board[r as usize][c as usize] == cell {
                    count += 1;
                } else {
                    break;
                }
            }
            // Backward
            for step in 1..4 {
                let r = row as i32 - dr * step;
                let c = col as i32 - dc * step;
                if r < 0 || r >= 6 || c < 0 || c >= 7 {
                    break;
                }
                if self.board[r as usize][c as usize] == cell {
                    count += 1;
                } else {
                    break;
                }
            }
            if count >= 4 {
                return true;
            }
        }
        false
    }

    fn has_winner(&self) -> bool {
        for row in 0..6 {
            for col in 0..7 {
                if self.board[row][col] != Cell::Empty && self.check_win(row, col) {
                    return true;
                }
            }
        }
        false
    }

    fn board_full(&self) -> bool {
        (0..7).all(|col| self.board[5][col] != Cell::Empty)
    }

    /// Encode board as 42-char string, row 5 (top) first, left to right.
    /// ' ' = empty, 'R' = Red, 'Y' = Yellow.
    fn board_string(&self) -> String {
        let mut s = String::with_capacity(42);
        for row in (0..6).rev() {
            for col in 0..7 {
                s.push(match self.board[row][col] {
                    Cell::Empty => ' ',
                    Cell::Red => 'R',
                    Cell::Yellow => 'Y',
                });
            }
        }
        s
    }

    /// Track last move for win detection optimization
    fn last_drop_row: Option<(usize, usize)> — skip, just scan
}

impl GameState for ConnectFour {
    type Move = CfMove;
    type Player = Player;
    type MoveList = Vec<CfMove>;

    fn current_player(&self) -> Player {
        self.current
    }

    fn available_moves(&self) -> Vec<CfMove> {
        if self.has_winner() {
            return vec![];
        }
        (0..7u8)
            .filter(|&col| self.drop_row(col as usize).is_some())
            .map(CfMove)
            .collect()
    }

    fn make_move(&mut self, mov: &CfMove) {
        let col = mov.0 as usize;
        if let Some(row) = self.drop_row(col) {
            self.board[row][col] = match self.current {
                Player::Red => Cell::Red,
                Player::Yellow => Cell::Yellow,
            };
        }
        self.current = match self.current {
            Player::Red => Player::Yellow,
            Player::Yellow => Player::Red,
        };
    }

    fn terminal_value(&self) -> Option<ProvenValue> {
        if self.has_winner() {
            // The winner just moved, so current player lost
            return Some(ProvenValue::Loss);
        }
        if self.board_full() {
            return Some(ProvenValue::Draw);
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Evaluator: heuristic based on center control + threats
// ---------------------------------------------------------------------------

struct CfEval;

impl Evaluator<CfConfig> for CfEval {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &ConnectFour,
        moves: &Vec<CfMove>,
        _: Option<SearchHandle<CfConfig>>,
    ) -> (Vec<()>, i64) {
        // Simple heuristic: count pieces in center columns
        let mut score: i64 = 0;
        for row in 0..6 {
            for col in 2..5 {
                match state.board[row][col] {
                    Cell::Red => score += 1,
                    Cell::Yellow => score -= 1,
                    Cell::Empty => {}
                }
            }
        }
        (vec![(); moves.len()], score)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match player {
            Player::Red => *evaln,
            Player::Yellow => -*evaln,
        }
    }

    fn evaluate_existing_state(
        &self, _: &ConnectFour, evaln: &i64, _: SearchHandle<CfConfig>,
    ) -> i64 {
        *evaln
    }
}

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

// ---------------------------------------------------------------------------
// WASM bindings
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ConnectFourWasm {
    manager: MCTSManager<CfConfig>,
}

#[wasm_bindgen]
impl ConnectFourWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            manager: MCTSManager::new(
                ConnectFour::new(), CfConfig, CfEval, UCTPolicy::new(1.0), (),
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
        let tree = types::export_tree::<CfConfig>(
            self.manager.tree().root_node(), max_depth, &|_| None,
        );
        serde_wasm_bindgen::to_value(&tree).unwrap()
    }

    /// 42-char string, top-row first, left-to-right. ' '=empty, 'R'=red, 'Y'=yellow.
    pub fn get_board(&self) -> String {
        self.manager.tree().root_state().board_string()
    }

    pub fn current_player(&self) -> String {
        format!("{:?}", self.manager.tree().root_state().current)
    }

    pub fn is_terminal(&self) -> bool {
        let s = self.manager.tree().root_state();
        s.has_winner() || s.board_full()
    }

    pub fn result(&self) -> String {
        let s = self.manager.tree().root_state();
        if s.has_winner() {
            // current player is the loser (winner just moved)
            match s.current {
                Player::Red => "Yellow".to_string(),
                Player::Yellow => "Red".to_string(),
            }
        } else if s.board_full() {
            "Draw".to_string()
        } else {
            String::new()
        }
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    pub fn apply_move(&mut self, col: &str) -> bool {
        if let Ok(c) = col.parse::<u8>() {
            if c < 7 {
                return self.manager.advance(&CfMove(c)).is_ok();
            }
        }
        false
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            ConnectFour::new(), CfConfig, CfEval, UCTPolicy::new(1.0), (),
        );
    }
}
```

**IMPORTANT:** Remove the invalid line `fn last_drop_row: Option<(usize, usize)> — skip, just scan` that I left as a comment — delete that line entirely.

- [ ] **Step 2: Register in lib.rs**

Add `mod connectfour;` and `pub use connectfour::ConnectFourWasm;` to `mcts-wasm/src/lib.rs`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p mcts-wasm`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add mcts-wasm/src/connectfour.rs mcts-wasm/src/lib.rs
git commit -m "feat: Connect Four WASM game with center-control heuristic"
```

---

### Task 4: Connect Four React Demo

**Files:**
- Create: `docs/src/components/demos/ConnectFourDemo.tsx`
- Create: `docs/src/components/demos/ConnectFourDemo.module.css`
- Modify: `docs/src/pages/playground.tsx`

- [ ] **Step 1: Create CSS module**

Create `docs/src/components/demos/ConnectFourDemo.module.css`:

```css
.container {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
}

.boardSection {
  flex: 0 0 auto;
}

.statsSection {
  flex: 1;
  min-width: 200px;
}

.status {
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}

.board {
  background: #1d4ed8;
  border-radius: 8px;
  padding: 8px;
  display: inline-block;
}

.grid {
  display: grid;
  grid-template-columns: repeat(7, 44px);
  grid-template-rows: repeat(6, 44px);
  gap: 4px;
}

.cell {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  transition: background 0.15s;
}

.cellEmpty {
  background: var(--ifm-background-color);
}

.cellEmpty:hover:not(:disabled) {
  background: var(--ifm-color-emphasis-200);
}

.cellRed {
  background: #ef4444;
  cursor: default;
}

.cellYellow {
  background: #eab308;
  cursor: default;
}

.cell:disabled {
  cursor: default;
}

.colHeaders {
  display: grid;
  grid-template-columns: repeat(7, 44px);
  gap: 4px;
  padding: 0 8px;
  margin-bottom: 4px;
}

.colHeader {
  text-align: center;
  font-size: 0.7rem;
  color: var(--ifm-color-emphasis-500);
  cursor: pointer;
  padding: 2px 0;
  border-radius: 4px;
}

.colHeader:hover {
  background: var(--ifm-color-emphasis-100);
}

.controls {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
}

.btn {
  padding: 0.375rem 0.75rem;
  font-size: 0.8rem;
  border: 1px solid var(--ifm-color-emphasis-300);
  border-radius: 4px;
  background: var(--ifm-background-surface-color);
  color: var(--ifm-font-color-base);
  cursor: pointer;
}

.btn:hover {
  background: var(--ifm-color-emphasis-100);
}

.gameOver {
  margin-top: 0.75rem;
  padding: 0.75rem;
  text-align: center;
  border-radius: 4px;
  background: var(--ifm-color-emphasis-100);
  font-weight: 600;
}

.moveList {
  font-size: 0.8rem;
  margin-top: 0.5rem;
}

.moveRow {
  display: flex;
  justify-content: space-between;
  padding: 0.2rem 0;
  border-bottom: 1px solid var(--ifm-color-emphasis-100);
}
```

- [ ] **Step 2: Create demo component**

Create `docs/src/components/demos/ConnectFourDemo.tsx`:

```tsx
import { useState, useEffect, useRef, useCallback } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './ConnectFourDemo.module.css';

interface ChildStat {
  mov: string;
  visits: number;
  avg_reward: number;
}

interface Stats {
  total_playouts: number;
  total_nodes: number;
  best_move?: string;
  children: ChildStat[];
}

const PLAYOUTS = 10000;

function ConnectFourDemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');
  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);
  const [board, setBoard] = useState(' '.repeat(42));
  const [currentPlayer, setCurrentPlayer] = useState('Red');
  const [stats, setStats] = useState<Stats | null>(null);
  const [gameResult, setGameResult] = useState('');
  const [thinking, setThinking] = useState(false);

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
    setCurrentPlayer(gameRef.current.current_player());
    setGameResult(gameRef.current.result());
    if (!gameRef.current.is_terminal()) {
      gameRef.current.playout_n(PLAYOUTS);
      setStats(gameRef.current.get_stats());
    }
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) gameRef.current.free();
    gameRef.current = new wasm.ConnectFourWasm();
    setThinking(false);
    setGameResult('');
    syncState();
  }, [wasm, syncState]);

  useEffect(() => {
    if (ready) initGame();
    return () => {
      if (gameRef.current) {
        gameRef.current.free();
        gameRef.current = null;
      }
    };
  }, [ready, initGame]);

  const handleColumnClick = useCallback((col: number) => {
    if (!gameRef.current || thinking || gameResult) return;

    gameRef.current.apply_move(col.toString());
    syncState();

    if (gameRef.current.result()) return;

    // MCTS response
    setThinking(true);
    setTimeout(() => {
      if (!gameRef.current) return;
      const best = gameRef.current.best_move();
      if (best) {
        gameRef.current.apply_move(best);
        syncState();
      }
      setThinking(false);
    }, 100);
  }, [thinking, gameResult, syncState]);

  if (error) return <div>Failed to load WASM: {error}</div>;
  if (!ready) return <div className={styles.status}>Loading WASM...</div>;

  // Parse 42-char board (top-row first, left-to-right)
  const cells = Array.from(board);

  return (
    <div className={styles.container}>
      <div className={styles.boardSection}>
        <div className={styles.status}>
          {gameResult
            ? null
            : thinking
              ? 'MCTS is thinking...'
              : `Your turn (Red) - click a column`}
        </div>

        <div className={styles.colHeaders}>
          {[0,1,2,3,4,5,6].map(col => (
            <div key={col} className={styles.colHeader}
              onClick={() => handleColumnClick(col)}>
              {'\u25BC'}
            </div>
          ))}
        </div>

        <div className={styles.board}>
          <div className={styles.grid}>
            {cells.map((cell, i) => (
              <button
                key={i}
                className={`${styles.cell} ${
                  cell === 'R' ? styles.cellRed :
                  cell === 'Y' ? styles.cellYellow :
                  styles.cellEmpty
                }`}
                onClick={() => handleColumnClick(i % 7)}
                disabled={thinking || !!gameResult || cell !== ' ' || currentPlayer !== 'Red'}
              />
            ))}
          </div>
        </div>

        {gameResult && (
          <div className={styles.gameOver}>
            {gameResult === 'Draw' ? "It's a draw!" :
             `${gameResult} wins!`}
          </div>
        )}

        <div className={styles.controls}>
          <button className={styles.btn} onClick={initGame}>New Game</button>
        </div>
      </div>

      <div className={styles.statsSection}>
        {stats && stats.children.length > 0 && (
          <div className={styles.moveList}>
            <div style={{ fontWeight: 600, marginBottom: '0.25rem', fontSize: '0.75rem' }}>
              MCTS Analysis ({stats.total_playouts.toLocaleString()} playouts)
            </div>
            {stats.children
              .filter(c => c.visits > 0)
              .sort((a, b) => b.visits - a.visits)
              .map(c => (
                <div key={c.mov} className={styles.moveRow}>
                  <span>Col {c.mov}</span>
                  <span>{c.visits} visits ({c.avg_reward.toFixed(1)})</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ConnectFourDemo() {
  return (
    <BrowserOnly fallback={<div className={styles.status}>Loading...</div>}>
      {() => {
        const { WasmProvider } = require('../mcts/WasmProvider');
        return (
          <WasmProvider>
            <ConnectFourDemoInner />
          </WasmProvider>
        );
      }}
    </BrowserOnly>
  );
}
```

- [ ] **Step 3: Add tab to playground**

Add `{ id: 'connectfour', label: 'Connect Four' }` to tabs and the corresponding DemoLoader case in `docs/src/pages/playground.tsx`.

- [ ] **Step 4: Verify build**

Run: `cd docs && npm run build`

- [ ] **Step 5: Commit**

```bash
git add docs/src/components/demos/ConnectFourDemo.tsx docs/src/components/demos/ConnectFourDemo.module.css docs/src/pages/playground.tsx
git commit -m "feat: Connect Four playground demo with MCTS analysis"
```

---

### Task 5: 2048 Rust + WASM

**Files:**
- Create: `mcts-wasm/src/game2048.rs`
- Modify: `mcts-wasm/src/lib.rs`

- [ ] **Step 1: Create the game module**

Create `mcts-wasm/src/game2048.rs`. This is the most complex game — it needs:
- 4x4 grid with slide/merge mechanics
- Random tile spawning (90% "2", 10% "4") as open-loop stochastic events
- Scoring based on merged tile values

```rust
use mcts::*;
use mcts::tree_policy::UCTPolicy;
use rand::prelude::*;
use rand::rngs::SmallRng;
use wasm_bindgen::prelude::*;
use serde_wasm_bindgen;
use serde::Serialize;

use crate::types;

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Game2048 {
    board: [[u32; 4]; 4],
    score: u32,
    rng: SmallRng,
}

#[derive(Clone, Debug, PartialEq)]
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

impl Game2048 {
    fn new() -> Self {
        let mut game = Self {
            board: [[0; 4]; 4],
            score: 0,
            rng: SmallRng::from_rng(rand::thread_rng()).unwrap(),
        };
        game.spawn_tile();
        game.spawn_tile();
        game
    }

    fn spawn_tile(&mut self) {
        let empty: Vec<(usize, usize)> = (0..4)
            .flat_map(|r| (0..4).map(move |c| (r, c)))
            .filter(|&(r, c)| self.board[r][c] == 0)
            .collect();
        if let Some(&(r, c)) = empty.choose(&mut self.rng) {
            self.board[r][c] = if self.rng.gen::<f64>() < 0.9 { 2 } else { 4 };
        }
    }

    /// Slide and merge a single row left. Returns (new_row, points_scored).
    fn slide_row(row: &[u32; 4]) -> ([u32; 4], u32) {
        let mut result = [0u32; 4];
        let mut score = 0u32;
        let mut pos = 0;
        let mut last_merged = false;

        for &val in row {
            if val == 0 {
                continue;
            }
            if pos > 0 && result[pos - 1] == val && !last_merged {
                result[pos - 1] *= 2;
                score += result[pos - 1];
                last_merged = true;
            } else {
                result[pos] = val;
                pos += 1;
                last_merged = false;
            }
        }
        (result, score)
    }

    /// Apply a direction, returning true if the board changed.
    fn apply_dir(&mut self, dir: &Dir) -> bool {
        let old_board = self.board;
        match dir {
            Dir::Left => {
                for r in 0..4 {
                    let (new_row, pts) = Self::slide_row(&self.board[r]);
                    self.board[r] = new_row;
                    self.score += pts;
                }
            }
            Dir::Right => {
                for r in 0..4 {
                    let mut rev = self.board[r];
                    rev.reverse();
                    let (mut new_row, pts) = Self::slide_row(&rev);
                    new_row.reverse();
                    self.board[r] = new_row;
                    self.score += pts;
                }
            }
            Dir::Up => {
                for c in 0..4 {
                    let col = [self.board[0][c], self.board[1][c], self.board[2][c], self.board[3][c]];
                    let (new_col, pts) = Self::slide_row(&col);
                    for r in 0..4 {
                        self.board[r][c] = new_col[r];
                    }
                    self.score += pts;
                }
            }
            Dir::Down => {
                for c in 0..4 {
                    let col = [self.board[3][c], self.board[2][c], self.board[1][c], self.board[0][c]];
                    let (new_col, pts) = Self::slide_row(&col);
                    for r in 0..4 {
                        self.board[3 - r][c] = new_col[r];
                    }
                    self.score += pts;
                }
            }
        }
        old_board != self.board
    }

    fn can_move(&self) -> bool {
        // Any empty cell
        for r in 0..4 {
            for c in 0..4 {
                if self.board[r][c] == 0 {
                    return true;
                }
            }
        }
        // Any adjacent equal
        for r in 0..4 {
            for c in 0..4 {
                let val = self.board[r][c];
                if r + 1 < 4 && self.board[r + 1][c] == val {
                    return true;
                }
                if c + 1 < 4 && self.board[r][c + 1] == val {
                    return true;
                }
            }
        }
        false
    }

    fn max_tile(&self) -> u32 {
        self.board.iter().flat_map(|r| r.iter()).copied().max().unwrap_or(0)
    }
}

impl GameState for Game2048 {
    type Move = Dir;
    type Player = ();
    type MoveList = Vec<Dir>;

    fn current_player(&self) -> Self::Player {}

    fn available_moves(&self) -> Vec<Dir> {
        if !self.can_move() {
            return vec![];
        }
        let mut moves = Vec::new();
        for dir in [Dir::Up, Dir::Down, Dir::Left, Dir::Right] {
            let mut test = self.clone();
            if test.apply_dir(&dir) {
                moves.push(dir);
            }
        }
        moves
    }

    fn make_move(&mut self, mov: &Dir) {
        if self.apply_dir(mov) {
            self.spawn_tile();
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator: score + empty cells + monotonicity
// ---------------------------------------------------------------------------

struct Eval2048;

impl Evaluator<Config2048> for Eval2048 {
    type StateEvaluation = i64;

    fn evaluate_new_state(
        &self,
        state: &Game2048,
        moves: &Vec<Dir>,
        _: Option<SearchHandle<Config2048>>,
    ) -> (Vec<()>, i64) {
        let empty_cells: i64 = state.board.iter()
            .flat_map(|r| r.iter())
            .filter(|&&v| v == 0)
            .count() as i64;
        let value = state.score as i64 + empty_cells * 10 + state.max_tile() as i64;
        (vec![(); moves.len()], value)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(
        &self, _: &Game2048, evaln: &i64, _: SearchHandle<Config2048>,
    ) -> i64 {
        *evaln
    }
}

#[derive(Default)]
struct Config2048;

impl MCTS for Config2048 {
    type State = Game2048;
    type Eval = Eval2048;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ();

    fn max_playout_depth(&self) -> usize {
        50
    }
}

// ---------------------------------------------------------------------------
// WASM bindings
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct Game2048Wasm {
    manager: MCTSManager<Config2048>,
}

#[wasm_bindgen]
impl Game2048Wasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            manager: MCTSManager::new(
                Game2048::new(), Config2048, Eval2048, UCTPolicy::new(1.0), (),
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

    /// Returns the 4x4 board as a flat JSON array of 16 numbers (row-major, top to bottom).
    pub fn get_board(&self) -> JsValue {
        let state = self.manager.tree().root_state();
        let flat: Vec<u32> = state.board.iter().flat_map(|r| r.iter().copied()).collect();
        serde_wasm_bindgen::to_value(&flat).unwrap()
    }

    pub fn score(&self) -> u32 {
        self.manager.tree().root_state().score
    }

    pub fn max_tile(&self) -> u32 {
        self.manager.tree().root_state().max_tile()
    }

    pub fn is_terminal(&self) -> bool {
        !self.manager.tree().root_state().can_move()
    }

    pub fn best_move(&self) -> Option<String> {
        self.manager.best_move().map(|m| format!("{m}"))
    }

    pub fn apply_move(&mut self, dir: &str) -> bool {
        let d = match dir {
            "Up" => Dir::Up,
            "Down" => Dir::Down,
            "Left" => Dir::Left,
            "Right" => Dir::Right,
            _ => return false,
        };
        self.manager.advance(&d).is_ok()
    }

    pub fn reset(&mut self) {
        self.manager = MCTSManager::new(
            Game2048::new(), Config2048, Eval2048, UCTPolicy::new(1.0), (),
        );
    }
}
```

- [ ] **Step 2: Register in lib.rs**

Add `mod game2048;` and `pub use game2048::Game2048Wasm;` to `mcts-wasm/src/lib.rs`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p mcts-wasm`
Expected: 0 errors. Note: `SmallRng` may need to be `rand_xoshiro::Xoshiro256PlusPlus` — check what the crate uses.

- [ ] **Step 4: Commit**

```bash
git add mcts-wasm/src/game2048.rs mcts-wasm/src/lib.rs
git commit -m "feat: 2048 WASM game with open-loop stochastic tile spawning"
```

---

### Task 6: 2048 React Demo

**Files:**
- Create: `docs/src/components/demos/Game2048Demo.tsx`
- Create: `docs/src/components/demos/Game2048Demo.module.css`
- Modify: `docs/src/pages/playground.tsx`

- [ ] **Step 1: Create CSS module**

Create `docs/src/components/demos/Game2048Demo.module.css`:

```css
.container {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
}

.boardSection {
  flex: 0 0 auto;
}

.statsSection {
  flex: 1;
  min-width: 200px;
}

.status {
  font-size: 0.875rem;
  margin-bottom: 0.75rem;
  font-weight: 600;
}

.scoreBar {
  display: flex;
  gap: 1rem;
  margin-bottom: 0.75rem;
  font-size: 0.875rem;
}

.scoreLabel {
  font-weight: 600;
}

.board {
  background: #bbada0;
  border-radius: 6px;
  padding: 8px;
  display: inline-block;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, 64px);
  grid-template-rows: repeat(4, 64px);
  gap: 6px;
}

.tile {
  width: 64px;
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
  font-weight: 700;
  font-size: 1.25rem;
  color: #776e65;
  background: #cdc1b4;
  transition: background 0.1s;
}

.tile[data-value="2"] { background: #eee4da; }
.tile[data-value="4"] { background: #ede0c8; }
.tile[data-value="8"] { background: #f2b179; color: #f9f6f2; }
.tile[data-value="16"] { background: #f59563; color: #f9f6f2; }
.tile[data-value="32"] { background: #f67c5f; color: #f9f6f2; }
.tile[data-value="64"] { background: #f65e3b; color: #f9f6f2; }
.tile[data-value="128"] { background: #edcf72; color: #f9f6f2; font-size: 1.1rem; }
.tile[data-value="256"] { background: #edcc61; color: #f9f6f2; font-size: 1.1rem; }
.tile[data-value="512"] { background: #edc850; color: #f9f6f2; font-size: 1.1rem; }
.tile[data-value="1024"] { background: #edc53f; color: #f9f6f2; font-size: 0.95rem; }
.tile[data-value="2048"] { background: #edc22e; color: #f9f6f2; font-size: 0.95rem; }

.controls {
  display: flex;
  gap: 0.5rem;
  margin-top: 1rem;
  flex-wrap: wrap;
}

.btn {
  padding: 0.375rem 0.75rem;
  font-size: 0.8rem;
  border: 1px solid var(--ifm-color-emphasis-300);
  border-radius: 4px;
  background: var(--ifm-background-surface-color);
  color: var(--ifm-font-color-base);
  cursor: pointer;
}

.btn:hover {
  background: var(--ifm-color-emphasis-100);
}

.btnPrimary {
  background: #8f7a66;
  color: #f9f6f2;
  border-color: #8f7a66;
}

.btnPrimary:hover {
  background: #9f8b77;
}

.gameOver {
  margin-top: 0.75rem;
  padding: 0.75rem;
  text-align: center;
  border-radius: 4px;
  background: var(--ifm-color-emphasis-100);
  font-weight: 600;
}

.moveList {
  font-size: 0.8rem;
  margin-top: 0.5rem;
}

.moveRow {
  display: flex;
  justify-content: space-between;
  padding: 0.2rem 0;
  border-bottom: 1px solid var(--ifm-color-emphasis-100);
}

.suggestion {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: var(--ifm-color-emphasis-100);
  border-radius: 4px;
  text-align: center;
  font-weight: 600;
}
```

- [ ] **Step 2: Create demo component**

Create `docs/src/components/demos/Game2048Demo.tsx`:

```tsx
import { useState, useEffect, useRef, useCallback } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './Game2048Demo.module.css';

interface ChildStat {
  mov: string;
  visits: number;
  avg_reward: number;
}

interface Stats {
  total_playouts: number;
  total_nodes: number;
  best_move?: string;
  children: ChildStat[];
}

const PLAYOUTS = 500;

function Game2048DemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');
  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);
  const [board, setBoard] = useState<number[]>(new Array(16).fill(0));
  const [score, setScore] = useState(0);
  const [maxTile, setMaxTile] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [gameOver, setGameOver] = useState(false);

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
    setScore(gameRef.current.score());
    setMaxTile(gameRef.current.max_tile());
    setGameOver(gameRef.current.is_terminal());
    if (!gameRef.current.is_terminal()) {
      gameRef.current.playout_n(PLAYOUTS);
      setStats(gameRef.current.get_stats());
    } else {
      setStats(null);
    }
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) gameRef.current.free();
    gameRef.current = new wasm.Game2048Wasm();
    syncState();
  }, [wasm, syncState]);

  useEffect(() => {
    if (ready) initGame();
    return () => {
      if (gameRef.current) {
        gameRef.current.free();
        gameRef.current = null;
      }
    };
  }, [ready, initGame]);

  const makeMove = useCallback((dir: string) => {
    if (!gameRef.current || gameOver) return;
    const ok = gameRef.current.apply_move(dir);
    if (ok) syncState();
  }, [gameOver, syncState]);

  const autoMove = useCallback(() => {
    if (!gameRef.current || gameOver || !stats?.best_move) return;
    gameRef.current.apply_move(stats.best_move);
    syncState();
  }, [gameOver, stats, syncState]);

  // Keyboard support
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const map: Record<string, string> = {
        ArrowUp: 'Up', ArrowDown: 'Down', ArrowLeft: 'Left', ArrowRight: 'Right',
        w: 'Up', s: 'Down', a: 'Left', d: 'Right',
      };
      if (map[e.key]) {
        e.preventDefault();
        makeMove(map[e.key]);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [makeMove]);

  if (error) return <div>Failed to load WASM: {error}</div>;
  if (!ready) return <div className={styles.status}>Loading WASM...</div>;

  return (
    <div className={styles.container}>
      <div className={styles.boardSection}>
        <div className={styles.scoreBar}>
          <span><span className={styles.scoreLabel}>Score:</span> {score.toLocaleString()}</span>
          <span><span className={styles.scoreLabel}>Best tile:</span> {maxTile}</span>
        </div>

        <div className={styles.board}>
          <div className={styles.grid}>
            {board.map((val, i) => (
              <div
                key={i}
                className={styles.tile}
                data-value={val > 0 ? Math.min(val, 2048) : undefined}
              >
                {val > 0 ? val : ''}
              </div>
            ))}
          </div>
        </div>

        {gameOver && (
          <div className={styles.gameOver}>
            Game Over! Score: {score.toLocaleString()}, Best tile: {maxTile}
          </div>
        )}

        <div className={styles.controls}>
          {['Up', 'Down', 'Left', 'Right'].map(dir => (
            <button key={dir} className={styles.btn} onClick={() => makeMove(dir)}
              disabled={gameOver}>
              {dir === 'Up' ? '\u2191' : dir === 'Down' ? '\u2193' : dir === 'Left' ? '\u2190' : '\u2192'} {dir}
            </button>
          ))}
          <button className={`${styles.btn} ${styles.btnPrimary}`} onClick={autoMove}
            disabled={gameOver || !stats?.best_move}>
            MCTS Move
          </button>
          <button className={styles.btn} onClick={initGame}>New Game</button>
        </div>

        {stats?.best_move && !gameOver && (
          <div className={styles.suggestion}>
            MCTS suggests: {stats.best_move}
          </div>
        )}
      </div>

      <div className={styles.statsSection}>
        {stats && stats.children.length > 0 && (
          <div className={styles.moveList}>
            <div style={{ fontWeight: 600, marginBottom: '0.25rem', fontSize: '0.75rem' }}>
              MCTS Analysis ({stats.total_playouts.toLocaleString()} playouts)
            </div>
            {stats.children
              .filter(c => c.visits > 0)
              .sort((a, b) => b.visits - a.visits)
              .map(c => (
                <div key={c.mov} className={styles.moveRow}>
                  <span>{c.mov}</span>
                  <span>{c.visits} visits (avg {c.avg_reward.toFixed(0)})</span>
                </div>
              ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function Game2048Demo() {
  return (
    <BrowserOnly fallback={<div className={styles.status}>Loading...</div>}>
      {() => {
        const { WasmProvider } = require('../mcts/WasmProvider');
        return (
          <WasmProvider>
            <Game2048DemoInner />
          </WasmProvider>
        );
      }}
    </BrowserOnly>
  );
}
```

- [ ] **Step 3: Add tab to playground**

Add `{ id: '2048', label: '2048' }` to tabs and the DemoLoader case in `docs/src/pages/playground.tsx`.

- [ ] **Step 4: Verify build**

Run: `cd docs && npm run build`

- [ ] **Step 5: Commit**

```bash
git add docs/src/components/demos/Game2048Demo.tsx docs/src/components/demos/Game2048Demo.module.css docs/src/pages/playground.tsx
git commit -m "feat: 2048 playground demo with MCTS move suggestions"
```

---

### Task 7: Final integration + WASM rebuild

**Files:**
- Verify: `mcts-wasm/src/lib.rs` (all 3 modules registered)
- Verify: `docs/src/pages/playground.tsx` (all 3 tabs added)

- [ ] **Step 1: Verify all Rust code compiles**

Run: `cargo check -p mcts-wasm`
Expected: 0 errors.

- [ ] **Step 2: Verify clippy**

Run: `cargo clippy -p mcts-wasm --all-targets`
Expected: 0 warnings.

- [ ] **Step 3: Verify docs build**

Run: `cd docs && npm run build`
Expected: Build succeeds.

- [ ] **Step 4: Build WASM package (if wasm-pack available)**

Run: `cd mcts-wasm && wasm-pack build --target web`
Expected: Package built in `mcts-wasm/pkg/`.

- [ ] **Step 5: Test playground locally (if wasm-pack succeeded)**

Run: `cd docs && npm start`
Visit: `http://localhost:3000/playground`
Expected: All 7 tabs visible (4 existing + 3 new). New games load and are playable.

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: final integration fixes for famous games playground"
```

---

## Verification

After all tasks:
```bash
cargo check -p mcts-wasm                      # all 3 new games compile
cargo clippy -p mcts-wasm --all-targets        # 0 warnings
cd docs && npm run build                       # docs site builds
```

If wasm-pack is available:
```bash
cd mcts-wasm && wasm-pack build --target web   # WASM package builds
cd docs && npm start                           # playground runs locally
```
