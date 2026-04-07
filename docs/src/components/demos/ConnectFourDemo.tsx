import { useCallback, useEffect, useRef, useState } from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './demos.module.css';
import boardStyles from './ConnectFourDemo.module.css';

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

type Phase = 'human' | 'mcts' | 'gameover';

function ConnectFourDemoInner() {
  const { useWasm } = require('../mcts/WasmProvider');

  const { wasm, ready, error } = useWasm();
  const gameRef = useRef<any>(null);
  const [board, setBoard] = useState<string>(' '.repeat(42));
  const [currentPlayer, setCurrentPlayer] = useState('Red');
  const [phase, setPhase] = useState<Phase>('human');
  const [stats, setStats] = useState<SearchStats | null>(null);
  const [winner, setWinner] = useState<string | null>(null);
  const [resultText, setResultText] = useState<string>('');

  const syncState = useCallback(() => {
    if (!gameRef.current) return;
    setBoard(gameRef.current.get_board());
    setCurrentPlayer(gameRef.current.current_player());
  }, []);

  const initGame = useCallback(() => {
    if (!wasm) return;
    if (gameRef.current) {
      gameRef.current.free();
    }
    gameRef.current = new wasm.ConnectFourWasm();
    setPhase('human');
    setStats(null);
    setWinner(null);
    setResultText('');
    syncState();
  }, [wasm, syncState]);

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

  const handleColumnClick = useCallback(
    (col: number) => {
      if (!gameRef.current || phase !== 'human') return;

      const success = gameRef.current.apply_move(col.toString());
      if (!success) return;

      syncState();

      if (gameRef.current.is_terminal()) {
        const result = gameRef.current.result();
        setPhase('gameover');
        if (result === 'Draw') {
          setWinner(null);
          setResultText("It's a draw!");
        } else {
          setWinner(result);
          setResultText(`${result} wins!`);
        }
        return;
      }

      // MCTS turn
      setPhase('mcts');

      setTimeout(() => {
        if (!gameRef.current) return;

        gameRef.current.playout_n(50000);
        const bestMove = gameRef.current.best_move();
        const s = gameRef.current.get_stats();
        setStats(s);

        if (bestMove != null) {
          gameRef.current.apply_move(bestMove);
          syncState();

          if (gameRef.current.is_terminal()) {
            const result = gameRef.current.result();
            setPhase('gameover');
            if (result === 'Draw') {
              setWinner(null);
              setResultText("It's a draw!");
            } else {
              setWinner(result);
              setResultText(`${result} wins!`);
            }
            return;
          }
        }

        setPhase('human');
      }, 100);
    },
    [phase, syncState],
  );

  if (error) {
    return <div className={styles.error}>Failed to load WASM: {error}</div>;
  }

  if (!ready) {
    return <div className={styles.loading}>Loading...</div>;
  }

  // Parse board string: 42 chars, top row first (row 5), left to right
  // chars 0-6 = top row, 7-13 = row 4, ..., 35-41 = bottom row (row 0)
  const cells: Array<' ' | 'R' | 'Y'> = [];
  for (let i = 0; i < 42; i++) {
    cells.push((board[i] || ' ') as ' ' | 'R' | 'Y');
  }

  const maxVisits = stats
    ? Math.max(...stats.children.map((c) => c.visits), 1)
    : 1;

  return (
    <div className={styles.demo}>
      <div className={styles.section}>
        <div className={boardStyles.board}>
          <div className={boardStyles.columnHeaders}>
            {Array.from({ length: 7 }, (_, col) => (
              <button
                key={col}
                className={boardStyles.columnButton}
                onClick={() => handleColumnClick(col)}
                disabled={phase !== 'human'}
                title={`Drop in column ${col + 1}`}
              >
                &#x25BC;
              </button>
            ))}
          </div>
          <div className={boardStyles.grid}>
            {cells.map((cell, i) => {
              const cellClass =
                cell === 'R'
                  ? boardStyles.cellRed
                  : cell === 'Y'
                    ? boardStyles.cellYellow
                    : boardStyles.cellEmpty;
              return (
                <div
                  key={i}
                  className={`${boardStyles.cell} ${cellClass}`}
                />
              );
            })}
          </div>
        </div>
      </div>

      <div className={styles.section}>
        {phase === 'human' && (
          <span
            className={boardStyles.statusBar}
          >
            <span className={boardStyles.turnRed}>Your turn (Red)</span>
            {' '}&#8212; click a column to drop a piece
          </span>
        )}
        {phase === 'mcts' && (
          <span className={styles.thinking}>MCTS is thinking...</span>
        )}
      </div>

      {phase === 'gameover' && (
        <div className={styles.gameOver}>
          <p>{resultText}</p>
          <button
            className="button button--sm button--outline button--primary"
            onClick={initGame}
          >
            New Game
          </button>
        </div>
      )}

      {stats && stats.children.length > 0 && (
        <div className={styles.section}>
          <div className={styles.sectionLabel}>
            MCTS analysis (Yellow / AI)
          </div>
          <table className={boardStyles.analysisTable}>
            <thead>
              <tr>
                <th>Column</th>
                <th>Visits</th>
                <th>Avg Reward</th>
                <th className={boardStyles.barCell}>Distribution</th>
              </tr>
            </thead>
            <tbody>
              {stats.children
                .slice()
                .sort((a, b) => parseInt(a.mov) - parseInt(b.mov))
                .map((child) => {
                  const isBest = stats.best_move === child.mov;
                  const pct = (child.visits / maxVisits) * 100;
                  return (
                    <tr key={child.mov}>
                      <td className={isBest ? boardStyles.bestCol : ''}>
                        {parseInt(child.mov) + 1}
                        {isBest ? ' *' : ''}
                      </td>
                      <td className={boardStyles.mono}>
                        {child.visits.toLocaleString()}
                      </td>
                      <td className={boardStyles.mono}>
                        {child.avg_reward.toFixed(3)}
                      </td>
                      <td className={boardStyles.barCell}>
                        <div
                          className={boardStyles.bar}
                          style={{ width: `${pct}%` }}
                        />
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
      )}

      <div className={styles.section} style={{ marginTop: '0.5rem' }}>
        <button
          className="button button--sm button--outline button--danger"
          onClick={initGame}
        >
          New Game
        </button>
      </div>
    </div>
  );
}

export default function ConnectFourDemo() {
  return (
    <BrowserOnly fallback={<div className={styles.loading}>Loading...</div>}>
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
