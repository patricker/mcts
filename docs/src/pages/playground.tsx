import { useState } from 'react';
import Layout from '@theme/Layout';
import BrowserOnly from '@docusaurus/BrowserOnly';
import styles from './playground.module.css';

const tabs = [
  { id: 'tictactoe', label: 'Tic-Tac-Toe' },
  { id: 'connectfour', label: 'Connect Four' },
  { id: '2048', label: '2048' },
  { id: 'nim', label: 'Nim' },
  { id: 'counting', label: 'Counting Game' },
  { id: 'dice', label: 'Dice Game' },
  { id: 'compare', label: 'Compare Policies' },
] as const;

type TabId = (typeof tabs)[number]['id'];

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

        <div className={styles.tabContent}>
          <BrowserOnly fallback={<div className={styles.loading}>Loading demo...</div>}>
            {() => <DemoLoader tab={activeTab} />}
          </BrowserOnly>
        </div>
      </main>
    </Layout>
  );
}
