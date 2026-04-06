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

const CELL_LABELS = ['top-left', 'top-center', 'top-right', 'mid-left', 'center', 'mid-right', 'bot-left', 'bot-center', 'bot-right'];

function TicTacToeDemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');

  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);
  const [board, setBoard] = useState('         ');
  const [phase, setPhase] = useState<Phase>('human');
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [provenValue, setProvenValue] = useState<string>('Unknown');
  const [gameResult, setGameResult] = useState<string>('');
  const [currentPlayer, setCurrentPlayer] = useState<string>('X');

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
    setCurrentPlayer(gameRef.current.current_player());
  }, []);

  const runAnalysis = useCallback(() => {
    if (!gameRef.current) return;
    gameRef.current.playout_n(5000);
    const s: SearchStats = gameRef.current.get_stats();
    setStats(s);
    setProvenValue(gameRef.current.root_proven_value());
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) {
      gameRef.current.free();
    }
    gameRef.current = new wasm.TicTacToeWasm();
    setPhase('human');
    setStats(null);
    setProvenValue('Unknown');
    setGameResult('');
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

  const handleCellClick = useCallback(
    (index: number) => {
      if (!gameRef.current || phase !== 'human') return;

      const currentBoard = gameRef.current.get_board();
      if (currentBoard[index] !== ' ') return;

      // Human plays X
      const success = gameRef.current.apply_move(String(index));
      if (!success) return;

      syncState();

      // Check if game is over after human move
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

        // Run MCTS for O
        gameRef.current.playout_n(5000);
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

        // Analyze human's new position
        runAnalysis();
        setPhase('human');
      }, 100);
    },
    [phase, syncState, runAnalysis],
  );

  if (error) {
    return <div className={sharedStyles.error}>Failed to load WASM: {error}</div>;
  }

  if (!ready) {
    return <div className={sharedStyles.loading}>Loading...</div>;
  }

  // Compute display labels for proven value.
  // root_proven_value() is from the current player's perspective.
  // Before MCTS moves, current player is O, so "Win" means O wins (MCTS wins).
  // When it's human's turn (X), current player is X, so "Win" means X wins (You win).
  let provenDisplay = provenValue;
  if (phase === 'human') {
    // Current player is X (human). Proven value is from X's perspective.
    if (provenValue === 'Win') provenDisplay = 'You win';
    else if (provenValue === 'Loss') provenDisplay = 'MCTS wins';
    else if (provenValue === 'Draw') provenDisplay = 'Draw';
  }

  // Game result display
  let resultMessage = '';
  if (phase === 'gameover') {
    if (gameResult === 'X') resultMessage = 'You win!';
    else if (gameResult === 'O') resultMessage = 'MCTS wins!';
    else if (gameResult === 'Draw') resultMessage = "It's a draw!";
  }

  // Status line
  let statusText = '';
  if (phase === 'human') statusText = 'Your turn (X)';
  else if (phase === 'mcts') statusText = 'MCTS is thinking...';

  const provenStatus = provenValue.toLowerCase() as 'win' | 'loss' | 'draw' | 'unknown';

  // For the analysis table, flip proven values since stats are from current player's
  // perspective (X when it's human's turn), but we want to show from human-friendly view.
  const displayChildren = stats?.children.map((c) => {
    // After human moves on cell c.mov, it becomes O's turn. The child's proven value
    // is from O's perspective in the tree. We display the cell index for clarity.
    return { ...c };
  }) ?? [];

  // Sort children by visits (descending)
  displayChildren.sort((a, b) => b.visits - a.visits);

  return (
    <div className={sharedStyles.demo}>
      {/* Status */}
      <div className={styles.status}>
        {phase === 'mcts' ? (
          <span className={sharedStyles.thinking}>MCTS is thinking...</span>
        ) : (
          statusText
        )}
      </div>

      {/* Board */}
      <div className={styles.board}>
        {board.split('').map((cell, i) => {
          const cellClasses = [
            styles.cell,
            cell === 'X' ? styles.cellX : '',
            cell === 'O' ? styles.cellO : '',
            (phase !== 'human' || cell !== ' ') ? styles.cellDisabled : '',
          ].filter(Boolean).join(' ');

          return (
            <button
              key={i}
              className={cellClasses}
              onClick={() => handleCellClick(i)}
              disabled={phase !== 'human' || cell !== ' '}
              aria-label={`Cell ${CELL_LABELS[i]}: ${cell === ' ' ? 'empty' : cell}`}
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
      {phase !== 'gameover' && (
        <div className={styles.controls}>
          <button
            className="button button--sm button--outline button--danger"
            onClick={initGame}
          >
            New Game
          </button>
        </div>
      )}

      {/* Proven value badge */}
      {phase !== 'gameover' && (
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
              {displayChildren.map((c) => (
                <tr key={c.mov}>
                  <td>{c.mov}</td>
                  <td>{c.visits.toLocaleString()}</td>
                  <td>{c.avg_reward.toFixed(3)}</td>
                  <td
                    className={styles.statusCell}
                    data-status={c.proven?.toLowerCase()}
                  >
                    {c.proven ?? '—'}
                  </td>
                </tr>
              ))}
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
