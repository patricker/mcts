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

const PLAYER_NAMES = ['X', 'O', 'A', 'B'];
const PLAYER_COLORS = ['#3b82f6', '#ef4444', '#22c55e', '#8b5cf6'];

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

  // Settings
  const [sCols, setSCols] = useState(3);
  const [sRows, setSRows] = useState(3);
  const [sK, setSK] = useState(3);
  const [sPlayers, setSPlayers] = useState(2);
  const [playouts, setPlayouts] = useState(5000);
  const [playerTypes, setPlayerTypes] = useState<string[]>(['human', 'mcts']);

  // Game state
  const [board, setBoard] = useState('');
  const [cols, setCols] = useState(3);
  const [rows, setRows] = useState(3);
  const [numPlayers, setNumPlayers] = useState(2);
  const [currentPlayer, setCurrentPlayer] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [resultText, setResultText] = useState('');
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [provenValue, setProvenValue] = useState('Unknown');
  const [thinking, setThinking] = useState(false);

  const playoutsRef = useRef(5000);
  const playerTypesRef = useRef(['human', 'mcts']);
  useEffect(() => { playoutsRef.current = playouts; }, [playouts]);
  useEffect(() => { playerTypesRef.current = playerTypes; }, [playerTypes]);

  useEffect(() => {
    setPlayerTypes((prev) =>
      Array.from({ length: sPlayers }, (_, i) => (i < prev.length ? prev[i] : 'mcts'))
    );
  }, [sPlayers]);

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
    setCurrentPlayer(gameRef.current.current_player());
    setGameOver(gameRef.current.is_terminal());
    const result = gameRef.current.result();
    if (result) {
      if (result === 'Draw') {
        setResultText("It's a draw!");
      } else {
        const pIdx = parseInt(result, 10) - 1;
        setResultText(`${PLAYER_NAMES[pIdx] ?? `Player ${result}`} wins!`);
      }
    } else {
      setResultText('');
    }
  }, []);

  const runAnalysis = useCallback(() => {
    if (!gameRef.current || gameRef.current.is_terminal()) {
      setStats(null);
      return;
    }
    gameRef.current.playout_n(playoutsRef.current);
    setStats(gameRef.current.get_stats());
    setProvenValue(gameRef.current.root_proven_value());
  }, []);

  const triggerMCTSTurn = useCallback(() => {
    setThinking(true);
    setTimeout(() => {
      if (!gameRef.current || gameRef.current.is_terminal()) {
        setThinking(false);
        syncState();
        return;
      }
      gameRef.current.playout_n(playoutsRef.current);
      const s = gameRef.current.get_stats();
      setStats(s);
      setProvenValue(gameRef.current.root_proven_value());
      const best = gameRef.current.best_move();
      if (best) {
        gameRef.current.apply_move(best);
        syncState();
      }
      setThinking(false);
      if (gameRef.current.is_terminal()) {
        setGameOver(true);
        syncState();
        return;
      }
      runAnalysis();
      const nextPlayer = gameRef.current.current_player();
      if (playerTypesRef.current[nextPlayer] === 'mcts') {
        setTimeout(() => triggerMCTSTurn(), 50);
      }
    }, 50);
  }, [syncState, runAnalysis]);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) gameRef.current.free();
    gameRef.current = new wasm.TicTacToeWasm(sCols, sRows, sK, sPlayers);
    setCols(gameRef.current.cols());
    setRows(gameRef.current.rows());
    setNumPlayers(gameRef.current.num_players());
    setThinking(false);
    setStats(null);
    setProvenValue('Unknown');
    setResultText('');
    setGameOver(false);
    syncState();
    runAnalysis();
    const firstPlayer = gameRef.current.current_player();
    if (playerTypesRef.current[firstPlayer] === 'mcts') {
      setTimeout(() => triggerMCTSTurn(), 100);
    }
  }, [wasm, sCols, sRows, sK, sPlayers, syncState, runAnalysis]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (ready) initGame();
    return () => {
      if (gameRef.current) {
        gameRef.current.free();
        gameRef.current = null;
      }
    };
  }, [ready]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCellClick = useCallback((index: number) => {
    if (!gameRef.current || gameOver || thinking) return;
    const cp = gameRef.current.current_player();
    if (playerTypesRef.current[cp] !== 'human') return;

    const currentBoard = gameRef.current.get_board();
    if (currentBoard[index] !== ' ') return;

    const success = gameRef.current.apply_move(String(index));
    if (!success) return;
    syncState();

    if (gameRef.current.is_terminal()) {
      setGameOver(true);
      syncState();
      return;
    }

    runAnalysis();
    const nextPlayer = gameRef.current.current_player();
    if (playerTypesRef.current[nextPlayer] === 'mcts') {
      triggerMCTSTurn();
    }
  }, [gameOver, thinking, syncState, runAnalysis, triggerMCTSTurn]);

  if (error) return <div className={sharedStyles.error}>Failed to load WASM: {error}</div>;
  if (!ready) return <div className={sharedStyles.loading}>Loading...</div>;

  const isHumanTurn = !gameOver && !thinking && playerTypes[currentPlayer] === 'human';
  const cellSize = cols <= 5 ? 56 : cols <= 7 ? 44 : 36;
  const fontSize = cols <= 5 ? '1.5rem' : cols <= 7 ? '1.1rem' : '0.9rem';
  const provenStatus = provenValue.toLowerCase() as 'win' | 'loss' | 'draw' | 'unknown';

  const displayChildren = (stats?.children ?? []).slice().sort((a, b) => b.visits - a.visits);

  return (
    <div className={sharedStyles.demo}>
      {/* Settings */}
      <div className={styles.settingsRow}>
        <label className={styles.settingLabel}>
          <span>Width</span>
          <select value={sCols} onChange={(e) => setSCols(Number(e.target.value))} className={styles.select}>
            {[2,3,4,5,6,7,8,9,10].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>Height</span>
          <select value={sRows} onChange={(e) => setSRows(Number(e.target.value))} className={styles.select}>
            {[2,3,4,5,6,7,8,9,10].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>In a Row</span>
          <select value={sK} onChange={(e) => setSK(Number(e.target.value))} className={styles.select}>
            {[2,3,4,5,6,7].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>Players</span>
          <select value={sPlayers} onChange={(e) => setSPlayers(Number(e.target.value))} className={styles.select}>
            {[2,3,4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className={styles.settingLabel}>
          <span>Playouts</span>
          <select value={playouts} onChange={(e) => setPlayouts(Number(e.target.value))} className={styles.select}>
            {PLAYOUT_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>
        <button
          className="button button--sm button--outline button--primary"
          onClick={initGame}
          style={{ alignSelf: 'flex-end', marginBottom: '0.2rem' }}
        >
          New Game
        </button>
      </div>

      {/* Player type toggles */}
      <div className={styles.settingsRow}>
        {Array.from({ length: sPlayers }, (_, i) => (
          <div key={i} style={{
            display: 'inline-flex', alignItems: 'center', gap: '0.375rem',
            padding: '0.25rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem',
            fontWeight: 600, border: '1px solid var(--ifm-color-emphasis-200)',
          }}>
            <span style={{
              display: 'inline-block', width: 14, height: 14, borderRadius: '50%',
              background: PLAYER_COLORS[i],
            }} />
            <span>{PLAYER_NAMES[i]}</span>
            <select
              value={playerTypes[i] || 'mcts'}
              onChange={(e) => {
                setPlayerTypes((prev) => {
                  const next = [...prev];
                  next[i] = e.target.value;
                  return next;
                });
              }}
              style={{ fontSize: '0.75rem', padding: '0.125rem 0.25rem' }}
            >
              <option value="human">Human</option>
              <option value="mcts">MCTS</option>
            </select>
          </div>
        ))}
      </div>

      <div className={styles.boardInfo}>
        {cols}x{rows}, {gameRef.current?.win_length() ?? sK}-in-a-row, {numPlayers} players
      </div>

      {/* Status */}
      <div className={styles.status}>
        {gameOver ? null : thinking ? (
          <span className={sharedStyles.thinking}>
            MCTS ({PLAYER_NAMES[currentPlayer]}) is thinking...
          </span>
        ) : isHumanTurn ? (
          <span>
            <span style={{ color: PLAYER_COLORS[currentPlayer] }}>{PLAYER_NAMES[currentPlayer]}</span>
            &apos;s turn — click a cell
          </span>
        ) : null}
      </div>

      {/* Board */}
      <div
        className={styles.board}
        style={{ gridTemplateColumns: `repeat(${cols}, ${cellSize}px)` }}
      >
        {board.split('').slice(0, cols * rows).map((cell, i) => {
          const pIdx = cell === ' ' ? -1 : PLAYER_NAMES.indexOf(cell);
          const color = pIdx >= 0 ? PLAYER_COLORS[pIdx] : undefined;
          const row = Math.floor(i / cols);
          const col = i % cols;

          return (
            <button
              key={i}
              className={`${styles.cell} ${cell !== ' ' || !isHumanTurn ? styles.cellDisabled : ''}`}
              style={{
                width: cellSize, height: cellSize, fontSize,
                color: color ?? 'inherit',
              }}
              onClick={() => handleCellClick(i)}
              disabled={!isHumanTurn || cell !== ' '}
              aria-label={`Row ${row + 1}, Col ${col + 1}: ${cell === ' ' ? 'empty' : cell}`}
            >
              {cell === ' ' ? '' : cell}
            </button>
          );
        })}
      </div>

      {/* Game over */}
      {gameOver && (
        <div className={sharedStyles.gameOver}>
          <p>{resultText}</p>
          <button className="button button--sm button--outline button--primary" onClick={initGame}>
            New Game
          </button>
        </div>
      )}

      {/* Proven value badge */}
      {!gameOver && provenValue !== 'Unknown' && (
        <div className={sharedStyles.section}>
          <span className={sharedStyles.provenValue} data-status={provenStatus}>
            Solver: {provenValue}
          </span>
        </div>
      )}

      {/* MCTS analysis — always shown */}
      {stats && displayChildren.length > 0 && (
        <div className={sharedStyles.section}>
          <div className={sharedStyles.sectionLabel}>
            MCTS analysis — {PLAYER_NAMES[currentPlayer]}&apos;s move ({stats.total_playouts.toLocaleString()} playouts, {stats.total_nodes.toLocaleString()} nodes)
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
