import { useCallback, useEffect, useRef, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import sharedStyles from './demos.module.css';
import styles from './Game2048Demo.module.css';

interface ChildStat {
  mov: string;
  visits: number;
  avg_reward: number;
}

interface SearchStats {
  total_playouts: number;
  total_nodes: number;
  best_move?: string;
  children: ChildStat[];
}

const DIRECTIONS = ['Up', 'Down', 'Left', 'Right'] as const;
type Direction = (typeof DIRECTIONS)[number];

const DIRECTION_ARROWS: Record<Direction, string> = {
  Up: '\u2191',
  Down: '\u2193',
  Left: '\u2190',
  Right: '\u2192',
};

const TILE_DATA_VALUES = [
  0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
  32768, 65536,
];

function tileDataValue(v: number): string {
  if (TILE_DATA_VALUES.includes(v)) return String(v);
  // For any value beyond our explicit CSS rules, use the largest bracket
  if (v > 2048) return '4096';
  return String(v);
}

function Game2048DemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');
  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);

  const [board, setBoard] = useState<number[]>(new Array(16).fill(0));
  const [score, setScore] = useState(0);
  const [maxTile, setMaxTile] = useState(0);
  const [terminal, setTerminal] = useState(false);
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [suggestion, setSuggestion] = useState<string | undefined>(undefined);

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    const b = gameRef.current.get_board();
    setBoard(Array.from(b));
    setScore(gameRef.current.score());
    setMaxTile(gameRef.current.max_tile());
    setTerminal(gameRef.current.is_terminal());
  }, []);

  const runAnalysis = useCallback(() => {
    if (!gameRef.current || gameRef.current.is_terminal()) {
      setStats(null);
      setSuggestion(undefined);
      return;
    }
    gameRef.current.playout_n(500);
    const s: SearchStats = gameRef.current.get_stats();
    setStats(s);
    setSuggestion(s.best_move ?? undefined);
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) {
      gameRef.current.free();
    }
    gameRef.current = new wasm.Game2048Wasm();
    syncState();
    runAnalysis();
  }, [wasm, syncState, runAnalysis]);

  useEffect(() => {
    if (ready) {
      initGame();
    }
    return () => {
      if (gameRef.current) {
        gameRef.current.free();
        gameRef.current = null;
      }
    };
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleMove = useCallback(
    (dir: Direction) => {
      if (!gameRef.current || terminal) return;
      const moved = gameRef.current.apply_move(dir);
      if (!moved) return;
      syncState();
      runAnalysis();
    },
    [terminal, syncState, runAnalysis],
  );

  const handleMCTSMove = useCallback(() => {
    if (!gameRef.current || terminal || !suggestion) return;
    gameRef.current.apply_move(suggestion);
    syncState();
    runAnalysis();
  }, [terminal, suggestion, syncState, runAnalysis]);

  // Keyboard support
  useEffect(() => {
    const keyMap: Record<string, Direction> = {
      ArrowUp: 'Up',
      ArrowDown: 'Down',
      ArrowLeft: 'Left',
      ArrowRight: 'Right',
      w: 'Up',
      W: 'Up',
      s: 'Down',
      S: 'Down',
      a: 'Left',
      A: 'Left',
      d: 'Right',
      D: 'Right',
    };

    function onKeyDown(e: KeyboardEvent) {
      const dir = keyMap[e.key];
      if (dir) {
        e.preventDefault();
        handleMove(dir);
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [handleMove]);

  if (error) {
    return (
      <div className={sharedStyles.error}>Failed to load WASM: {error}</div>
    );
  }

  if (!ready) {
    return <div className={sharedStyles.loading}>Loading...</div>;
  }

  // Build 4x4 grid from flat array
  const grid: number[][] = [];
  for (let r = 0; r < 4; r++) {
    grid.push(board.slice(r * 4, r * 4 + 4));
  }

  return (
    <div className={sharedStyles.demo}>
      {/* Score row */}
      <div className={styles.boardContainer}>
        <div className={styles.scoreRow}>
          <div className={styles.scoreBadge}>
            <span>Score</span>
            <span>{score}</span>
          </div>
          <div className={styles.scoreBadge}>
            <span>Best tile</span>
            <span>{maxTile}</span>
          </div>
        </div>

        {/* Board */}
        <div className={styles.board}>
          {grid.flat().map((val, i) => (
            <div
              key={i}
              className={styles.tile}
              data-value={tileDataValue(val)}
            >
              {val > 0 ? val : ''}
            </div>
          ))}
          {terminal && (
            <div className={styles.gameOverOverlay}>
              <p>Game Over!</p>
              <button
                className="button button--sm button--primary"
                onClick={initGame}
              >
                New Game
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Arrow controls */}
      <div className={styles.controls}>
        <div className={styles.controlRow}>
          <div className={styles.arrowSpacer} />
          <button
            className={styles.arrowBtn}
            onClick={() => handleMove('Up')}
            disabled={terminal}
            title="Up (W / Arrow Up)"
          >
            {DIRECTION_ARROWS.Up}
          </button>
          <div className={styles.arrowSpacer} />
        </div>
        <div className={styles.controlRow}>
          <button
            className={styles.arrowBtn}
            onClick={() => handleMove('Left')}
            disabled={terminal}
            title="Left (A / Arrow Left)"
          >
            {DIRECTION_ARROWS.Left}
          </button>
          <button
            className={styles.arrowBtn}
            onClick={() => handleMove('Down')}
            disabled={terminal}
            title="Down (S / Arrow Down)"
          >
            {DIRECTION_ARROWS.Down}
          </button>
          <button
            className={styles.arrowBtn}
            onClick={() => handleMove('Right')}
            disabled={terminal}
            title="Right (D / Arrow Right)"
          >
            {DIRECTION_ARROWS.Right}
          </button>
        </div>
        <div className={styles.keyHint}>Arrow keys or WASD</div>
        <div className={styles.actionBtns}>
          <button
            className="button button--sm button--outline button--primary"
            onClick={handleMCTSMove}
            disabled={terminal || !suggestion}
          >
            MCTS Move
          </button>
          <button
            className="button button--sm button--outline button--danger"
            onClick={initGame}
          >
            New Game
          </button>
        </div>
      </div>

      {/* MCTS suggestion */}
      {suggestion && !terminal && (
        <div className={styles.suggestion}>
          MCTS suggests: {DIRECTION_ARROWS[suggestion as Direction]}{' '}
          {suggestion}
        </div>
      )}

      {/* MCTS analysis table */}
      {stats && stats.children.length > 0 && (
        <div className={sharedStyles.section}>
          <div className={sharedStyles.sectionLabel}>
            MCTS analysis ({stats.total_playouts.toLocaleString()} playouts,{' '}
            {stats.total_nodes.toLocaleString()} nodes)
          </div>
          <table className={styles.analysisTable}>
            <thead>
              <tr>
                <th>Direction</th>
                <th>Visits</th>
                <th>Avg reward</th>
              </tr>
            </thead>
            <tbody>
              {stats.children
                .slice()
                .sort((a, b) => b.visits - a.visits)
                .map((c) => (
                  <tr
                    key={c.mov}
                    className={
                      c.mov === stats.best_move ? styles.bestRow : undefined
                    }
                  >
                    <td>
                      {DIRECTION_ARROWS[c.mov as Direction] ?? ''} {c.mov}
                    </td>
                    <td>{c.visits.toLocaleString()}</td>
                    <td>{c.avg_reward.toFixed(2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function Game2048Demo() {
  return (
    <BrowserOnly
      fallback={<div className={sharedStyles.loading}>Loading...</div>}
    >
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
