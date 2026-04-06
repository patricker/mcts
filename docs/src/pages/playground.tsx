import { useState } from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './playground.module.css';

interface TabInfo {
  id: string;
  label: string;
  description: string;
  concepts: string;
}

const tabs: TabInfo[] = [
  {
    id: 'tictactoe',
    label: 'Tic-Tac-Toe',
    description: 'Play against MCTS with the solver enabled. Watch it prove that perfect play leads to a draw \u2014 every position is classified as Win, Loss, or Draw.',
    concepts: 'Two-player games, MCTS-Solver, proven values',
  },
  {
    id: 'connectfour',
    label: 'Connect Four',
    description: 'Challenge MCTS to Connect Four. With 10,000 playouts per move, it evaluates every column and picks the strongest. Can you find a weakness?',
    concepts: 'Deep search, heuristic evaluation, exploration vs exploitation',
  },
  {
    id: '2048',
    label: '2048',
    description: 'MCTS suggests moves in 2048 by simulating hundreds of random futures. The random tile spawns make this a stochastic game \u2014 MCTS handles uncertainty naturally.',
    concepts: 'Stochastic games, open-loop chance nodes, depth-limited search',
  },
  {
    id: 'nim',
    label: 'Nim',
    description: 'A classic combinatorial game. MCTS-Solver proves every Nim position \u2014 take 1 or 2 stones, and the solver tells you exactly who wins.',
    concepts: 'Solver, game theory, terminal values',
  },
  {
    id: 'counting',
    label: 'Counting Game',
    description: 'The simplest possible MCTS example. Watch the tree grow as search discovers that incrementing toward 100 is better than decrementing.',
    concepts: 'Tree growth, visit allocation, basic MCTS',
  },
  {
    id: 'dice',
    label: 'Dice Game',
    description: 'Roll or stop \u2014 a simple stochastic game with chance nodes. Each die roll creates a branch in the search tree.',
    concepts: 'Chance nodes, expected value, risk assessment',
  },
  {
    id: 'compare',
    label: 'Compare Policies',
    description: 'See UCT vs PUCT side by side. PUCT uses prior probabilities to guide search, while UCT treats all moves equally until visited.',
    concepts: 'UCT, PUCT, neural network priors, AlphaGoPolicy',
  },
];

type TabId = TabInfo['id'];

function DemoLoader({ tab }: { tab: TabId }) {
  switch (tab) {
    case 'tictactoe': {
      const TicTacToeDemo =
        require('@site/src/components/demos/TicTacToeDemo').default;
      return <TicTacToeDemo />;
    }
    case 'connectfour': {
      const ConnectFourDemo =
        require('@site/src/components/demos/ConnectFourDemo').default;
      return <ConnectFourDemo />;
    }
    case '2048': {
      const Game2048Demo =
        require('@site/src/components/demos/Game2048Demo').default;
      return <Game2048Demo />;
    }
    case 'counting': {
      const StepThroughDemo =
        require('@site/src/components/demos/StepThroughDemo').default;
      return <StepThroughDemo />;
    }
    case 'nim': {
      const NimSolverDemo =
        require('@site/src/components/demos/NimSolverDemo').default;
      return <NimSolverDemo />;
    }
    case 'dice': {
      const ChanceNodesDemo =
        require('@site/src/components/demos/ChanceNodesDemo').default;
      return <ChanceNodesDemo />;
    }
    case 'compare': {
      const UCTvsPUCTDemo =
        require('@site/src/components/demos/UCTvsPUCTDemo').default;
      return <UCTvsPUCTDemo />;
    }
  }
}

export default function Playground(): JSX.Element {
  const [activeTab, setActiveTab] = useState<TabId>('tictactoe');

  return (
    <Layout
      title="Playground"
      description="Interactive MCTS demos powered by WASM"
    >
      <main className={styles.playground}>
        <div className={styles.header}>
          <h1>Playground</h1>
          <p>Interactive demos running the actual MCTS library via WebAssembly.</p>
        </div>

        <div className={styles.tabBar}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`${styles.tab} ${activeTab === tab.id ? styles.tabActive : ''}`}
              onClick={() => setActiveTab(tab.id)}
              type="button"
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className={styles.tabDescription}>
          <p>{tabs.find(t => t.id === activeTab)?.description}</p>
          <span className={styles.conceptsLabel}>
            Concepts: {tabs.find(t => t.id === activeTab)?.concepts}
          </span>
        </div>

        <div className={styles.tabContent}>
          <BrowserOnly fallback={<div className={styles.loading}>Loading demo...</div>}>
            {() => <DemoLoader tab={activeTab} />}
          </BrowserOnly>
        </div>
      </main>
    </Layout>
  );
}
