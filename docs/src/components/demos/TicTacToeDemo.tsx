import { useCallback, useEffect, useRef, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import sharedStyles from './demos.module.css';
import styles from './TicTacToeDemo.module.css';

interface ChildStat {
  mov: string;
  visits: number;
  avg_reward: number;
  proven?: string;
}

interface SearchStats {
  total_playouts: number;
  total_nodes: number;
  best_move?: string;
  children: ChildStat[];
}

type Phase = 'human' | 'mcts' | 'gameover';

const PLAYOUT_OPTIONS = [
  { label: '500', value: 500 },
  { label: '2,000', value: 2000 },
  { label: '5,000', value: 5000 },
  { label: '20,000', value: 20000 },
];

function TicTacToeDemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');

  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);

  // Settings (applied on New Game)
  const [settingCols, setSettingCols] = useState(3);
  const [settingRows, setSettingRows] = useState(3);
  const [settingK, setSettingK] = useState(3);
  const [playouts, setPlayouts] = useState(5000);

  // Game state
  const [board, setBoard] = useState('         ');
  const [cols, setCols] = useState(3);
  const [rows, setRows] = useState(3);
  const [k, setK] = useState(3);
  const [phase, setPhase] = useState<Phase>('human');
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [provenValue, setProvenValue] = useState<string>('Unknown');
  const [gameResult, setGameResult] = useState<string>('');

  const playoutsRef = useRef(5000);
  useEffect(() => { playoutsRef.current = playouts; }, [playouts]);

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
  }, []);

  const runAnalysis = useCallback(() => {
    if (!gameRef.current || gameRef.current.is_terminal()) return;
    gameRef.current.playout_n(playoutsRef.current);
    const s: SearchStats = gameRef.current.get_stats();
    setStats(s);
    setProvenValue(gameRef.current.root_proven_value());
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) {
      gameRef.current.free();
    }
    const c = settingCols;
    const r = settingRows;
    const kk = settingK;
    gameRef.current = new wasm.TicTacToeWasm(c, r, kk);
    setCols(gameRef.current.cols());
    setRows(gameRef.current.rows());
    setK(gameRef.current.win_length());
    setPhase('human');
    setStats(null);
    setProvenValue('Unknown');
    setGameResult('');
    syncState();
    runAnalysis();
  }, [wasm, settingCols, settingRows, settingK, syncState, runAnalysis]);

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

  const handleCellClick = useCallback(
    (index: number) => {
      if (!gameRef.current || phase !== 'human') return;

      const currentBoard = gameRef.current.get_board();
      if (currentBoard[index] !== ' ') return;

      const success = gameRef.current.apply_move(String(index));
      if (!success) return;

      syncState();

      if (gameRef.current.is_terminal()) {
        const result = gameRef.current.result();
        setGameResult(result);
        setPhase('gameover');
        setStats(null);
        setProvenValue('Unknown');
        return;
      }

      // MCTS turn
      setPhase('mcts');

      setTimeout(() => {
        if (!gameRef.current) return;

        gameRef.current.playout_n(playoutsRef.current);
        const bestMove = gameRef.current.best_move();

        if (bestMove) {
          gameRef.current.apply_move(bestMove);
          syncState();

          if (gameRef.current.is_terminal()) {
            const result = gameRef.current.result();
            setGameResult(result);
            setPhase('gameover');
            setStats(null);
            setProvenValue('Unknown');
            return;
          }
        }

        runAnalysis();
        setPhase('human');
      }, 50);
    },
    [phase, syncState, runAnalysis],
  );

  if (error) {
    return <div className={sharedStyles.error}>Failed to load WASM: {error}</div>;
  }

  if (!ready) {
    return <div className={sharedStyles.loading}>Loading...</div>;
  }

  // Proven value display (from current player's perspective)
  let provenDisplay = provenValue;
  if (phase === 'human') {
    if (provenValue === 'Win') provenDisplay = 'You win';
    else if (provenValue === 'Loss') provenDisplay = 'MCTS wins';
    else if (provenValue === 'Draw') provenDisplay = 'Draw';
  }

  let resultMessage = '';
  if (phase === 'gameover') {
    if (gameResult === 'X') resultMessage = 'You win!';
    else if (gameResult === 'O') resultMessage = 'MCTS wins!';
    else if (gameResult === 'Draw') resultMessage = "It's a draw!";
  }

  const provenStatus = provenValue.toLowerCase() as 'win' | 'loss' | 'draw' | 'unknown';

  const displayChildren = (stats?.children ?? [])
    .slice()
    .sort((a, b) => b.visits - a.visits);

  const cellSize = cols <= 5 ? 56 : cols <= 7 ? 44 : 36;
  const fontSize = cols <= 5 ? '1.5rem' : cols <= 7 ? '1.1rem' : '0.9rem';

  return (
    <div className={sharedStyles.demo}>
      {/* Settings */}
      <div className={styles.settingsRow}>
        <label className={styles.settingLabel}>
          <span>Width</span>
          <select
            value={settingCols}
            onChange={(e) => setSettingCols(Number(e.target.value))}
            className={styles.select}
          >
            {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>Height</span>
          <select
            value={settingRows}
            onChange={(e) => setSettingRows(Number(e.target.value))}
            className={styles.select}
          >
            {[2, 3, 4, 5, 6, 7, 8, 9, 10].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>In a Row</span>
          <select
            value={settingK}
            onChange={(e) => setSettingK(Number(e.target.value))}
            className={styles.select}
          >
            {[2, 3, 4, 5, 6, 7].map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>Playouts</span>
          <select
            value={playouts}
            onChange={(e) => setPlayouts(Number(e.target.value))}
            className={styles.select}
          >
            {PLAYOUT_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </label>
      </div>

      <div className={styles.boardInfo}>
        {cols}x{rows}, {k}-in-a-row
      </div>

      {/* Status */}
      <div className={styles.status}>
        {phase === 'mcts' ? (
          <span className={sharedStyles.thinking}>MCTS is thinking...</span>
        ) : phase === 'human' ? (
          'Your turn (X)'
        ) : null}
      </div>

      {/* Board */}
      <div
        className={styles.board}
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`,
        }}
      >
        {board.split('').slice(0, cols * rows).map((cell, i) => {
          const cellClasses = [
            styles.cell,
            cell === 'X' ? styles.cellX : '',
            cell === 'O' ? styles.cellO : '',
            (phase !== 'human' || cell !== ' ') ? styles.cellDisabled : '',
          ].filter(Boolean).join(' ');

          const row = Math.floor(i / cols);
          const col = i % cols;

          return (
            <button
              key={i}
              className={cellClasses}
              style={{ width: cellSize, height: cellSize, fontSize }}
              onClick={() => handleCellClick(i)}
              disabled={phase !== 'human' || cell !== ' '}
              aria-label={`Row ${row + 1}, Col ${col + 1}: ${cell === ' ' ? 'empty' : cell}`}
            >
              {cell === ' ' ? '' : cell}
            </button>
          );
        })}
      </div>

      {/* Game over */}
      {phase === 'gameover' && (
        <div className={sharedStyles.gameOver}>
          <p>{resultMessage}</p>
          <button
            className="button button--sm button--outline button--primary"
            onClick={initGame}
          >
            New Game
          </button>
        </div>
      )}

      {/* Controls */}
      <div className={styles.controls}>
        <button
          className="button button--sm button--outline button--danger"
          onClick={initGame}
        >
          New Game
        </button>
      </div>

      {/* Proven value badge */}
      {phase !== 'gameover' && provenValue !== 'Unknown' && (
        <div className={sharedStyles.section}>
          <span className={sharedStyles.provenValue} data-status={provenStatus}>
            Solver: {provenDisplay}
          </span>
        </div>
      )}

      {/* MCTS analysis */}
      {stats && displayChildren.length > 0 && (
        <div className={sharedStyles.section}>
          <div className={sharedStyles.sectionLabel}>
            MCTS analysis ({stats.total_playouts.toLocaleString()} playouts, {stats.total_nodes.toLocaleString()} nodes)
          </div>
          <table className={styles.analysisTable}>
            <thead>
              <tr>
                <th>Cell</th>
                <th>Visits</th>
                <th>Avg Reward</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {displayChildren.slice(0, 15).map((c) => {
                const ci = parseInt(c.mov, 10);
                const cr = Math.floor(ci / cols);
                const cc = ci % cols;
                return (
                  <tr key={c.mov}>
                    <td>({cr + 1},{cc + 1})</td>
                    <td>{c.visits.toLocaleString()}</td>
                    <td>{c.avg_reward.toFixed(3)}</td>
                    <td
                      className={styles.statusCell}
                      data-status={c.proven?.toLowerCase()}
                    >
                      {c.proven ?? '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function TicTacToeDemo() {
  return (
    <BrowserOnly fallback={<div className={sharedStyles.loading}>Loading...</div>}>
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
