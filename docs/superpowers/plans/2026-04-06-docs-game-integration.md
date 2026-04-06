# Docs Game Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Weave Tic-Tac-Toe, Connect Four, and 2048 into the tutorial narrative so each game teaches a specific MCTS concept, and add educational context to the playground page.

**Architecture:** Edit existing tutorial MDX files to add game references, embed playground demos where relevant, and add per-game descriptions to the playground page. No new pages — augment existing content.

**Tech Stack:** MDX (Docusaurus), React component imports

---

## File Structure

| File | Change |
|------|--------|
| `docs/docs/tutorials/03-two-player-games.md` | Add TTT as second example, link to playground |
| `docs/docs/tutorials/04-solving-games.md` | Add TTT solver showcase, proven-draw discussion |
| `docs/docs/tutorials/05-stochastic-games.md` | Add 2048 as primary stochastic example |
| `docs/docs/concepts/chance-nodes.md` | Add 2048 examples in open-loop section |
| `docs/src/pages/playground.tsx` | Add per-game descriptions explaining what each teaches |
| `docs/docs/intro.md` | Mention famous games in learning path |

---

### Task 1: Playground educational descriptions

**Files:**
- Modify: `docs/src/pages/playground.tsx`

- [ ] **Step 1: Add descriptions to the tabs data structure**

Read `docs/src/pages/playground.tsx`. Update the `tabs` array to include a `description` field for each tab, and render it below the tab bar:

```tsx
const tabs = [
  {
    id: 'tictactoe',
    label: 'Tic-Tac-Toe',
    description: 'Play against MCTS with the solver enabled. Watch it prove that perfect play leads to a draw — every position is classified as Win, Loss, or Draw.',
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
    description: 'MCTS suggests moves in 2048 by simulating hundreds of random futures. The random tile spawns make this a stochastic game — MCTS handles uncertainty naturally.',
    concepts: 'Stochastic games, open-loop chance nodes, depth-limited search',
  },
  {
    id: 'nim',
    label: 'Nim',
    description: 'A classic combinatorial game. MCTS-Solver proves every Nim position — take 1 or 2 stones, and the solver tells you exactly who wins.',
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
    description: 'Roll or stop — a simple stochastic game with chance nodes. Each die roll creates a branch in the search tree.',
    concepts: 'Chance nodes, expected value, risk assessment',
  },
  {
    id: 'compare',
    label: 'Compare Policies',
    description: 'See UCT vs PUCT side by side. PUCT uses prior probabilities to guide search, while UCT treats all moves equally until visited.',
    concepts: 'UCT, PUCT, neural network priors, AlphaGoPolicy',
  },
] as const;
```

Add a description section below the tab bar that shows the active tab's description and concepts:

```tsx
<div className={styles.tabDescription}>
  <p>{tabs.find(t => t.id === activeTab)?.description}</p>
  <span className={styles.conceptsLabel}>
    Concepts: {tabs.find(t => t.id === activeTab)?.concepts}
  </span>
</div>
```

Add corresponding CSS to `docs/src/pages/playground.module.css`:

```css
.tabDescription {
  padding: 0.75rem 1rem;
  margin-bottom: 1rem;
  background: var(--ifm-color-emphasis-100);
  border-radius: 6px;
  font-size: 0.875rem;
}

.tabDescription p {
  margin: 0 0 0.25rem 0;
}

.conceptsLabel {
  font-size: 0.75rem;
  color: var(--ifm-color-emphasis-600);
  font-style: italic;
}
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`
Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add docs/src/pages/playground.tsx docs/src/pages/playground.module.css
git commit -m "docs: add educational descriptions to playground tabs"
```

---

### Task 2: Tutorial 3 — Tic-Tac-Toe as two-player example

**Files:**
- Modify: `docs/docs/tutorials/03-two-player-games.md`

- [ ] **Step 1: Read the current tutorial and add TTT references**

Read `docs/docs/tutorials/03-two-player-games.md` in full. Add content at these insertion points:

**At the end of the intro section** (after the Nim description), add a paragraph:

```markdown
:::tip Try it yourself
While we use Nim as our teaching example (it's simpler to implement), the same concepts apply to any two-player game. **[Play Tic-Tac-Toe against MCTS in the Playground →](/playground)** — the solver proves every position as Win, Loss, or Draw in real time.
:::
```

**At the end of the tutorial** (before "What's next"), add a section:

```markdown
## Beyond Nim: Tic-Tac-Toe and Connect Four

Everything you learned here — player alternation, negamax evaluation, `terminal_value()` — works identically for Tic-Tac-Toe and Connect Four. The only differences are board representation and win detection:

- **Tic-Tac-Toe**: 9 cells, 8 winning lines, branching factor ≤ 9. Small enough to solve completely — every position is provably Win, Loss, or Draw.
- **Connect Four**: 42 cells, gravity-based drops, 4-in-a-row detection. Much deeper trees — MCTS plays strong but can't fully solve it in a browser.

Both are available in the [Playground](/playground). Try them and watch how MCTS allocates visits differently based on position quality — the same negamax evaluation you just learned, working on real games.
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/tutorials/03-two-player-games.md
git commit -m "docs: add TTT and Connect Four references to two-player tutorial"
```

---

### Task 3: Tutorial 4 — Tic-Tac-Toe solver showcase

**Files:**
- Modify: `docs/docs/tutorials/04-solving-games.md`

- [ ] **Step 1: Read the current tutorial and add TTT solver content**

Read `docs/docs/tutorials/04-solving-games.md` in full. Add content at these insertion points:

**After explaining how the solver propagates proven values** (after the ProvenValue table), add:

```markdown
### Solving Tic-Tac-Toe

Tic-Tac-Toe is the ideal solver showcase. With only ~5,478 distinct game states, MCTS-Solver proves the entire game tree within a few thousand playouts:

- From an empty board: **Proven Draw** (both players can force a draw with optimal play)
- After X plays center: **Proven Draw** (O can always equalize)
- After X plays corner, O plays non-center: **Proven Win for X** (X has a forced win)

**[Try it in the Playground →](/playground)** — play as X and watch MCTS prove each position. The solver badge shows Win/Loss/Draw for the current position, and each move shows its proven value. Notice how MCTS stops searching once a position is fully resolved.

This is exactly what happens in [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) during endgame play — the solver kicks in when the remaining tree is small enough to prove, saving computation and guaranteeing optimal moves.
```

**In the "When to enable" section**, add a note:

```markdown
:::note Games in the Playground
Tic-Tac-Toe has solver enabled — positions are proven in real time. Connect Four does not — the tree is too deep for browser-speed proofs, so it relies on heuristic evaluation. This illustrates the practical boundary: solver is powerful but needs a manageable game tree.
:::
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/tutorials/04-solving-games.md
git commit -m "docs: add TTT solver showcase to solving tutorial"
```

---

### Task 4: Tutorial 5 — 2048 as stochastic example

**Files:**
- Modify: `docs/docs/tutorials/05-stochastic-games.md`

- [ ] **Step 1: Read the current tutorial and add 2048 content**

Read `docs/docs/tutorials/05-stochastic-games.md` in full. Add content at these insertion points:

**At the end of the intro** (after the dice game description), add:

```markdown
:::tip A game you already know
The dice game teaches the concept, but **2048** is where it clicks. After every slide, the game randomly places a 2 (90%) or 4 (10%) on an empty tile. MCTS handles this by simulating hundreds of possible futures — different tile placements, different move sequences — and recommending the direction that leads to the highest average score. **[Try 2048 in the Playground →](/playground)**
:::
```

**After the open-loop vs closed-loop explanation**, add a section:

```markdown
## 2048: Open-Loop in Action

2048 is a natural fit for open-loop MCTS. Here's why:

1. **The player decides before the RNG fires.** You choose Up/Down/Left/Right, *then* a random tile appears. Your decision can't depend on which tile appears next.
2. **The outcome space is huge.** After a move, a 2 or 4 could appear in any empty cell. With 10 empty cells, that's 20 possible outcomes. Closed-loop would create 20 child nodes per chance event — memory explodes.
3. **Averaging works well.** Open-loop samples different tile placements across playouts and averages the results. After 500 playouts, the estimate of "how good is sliding Left?" is reliable regardless of which specific tile appears.

This is the same reason real 2048 AIs use expectimax (averaging over chance) rather than minimax — the randomness is something you plan *around*, not *against*.

In the Playground, watch the MCTS analysis panel: each direction shows its average reward across hundreds of simulated futures. The suggested move is the direction with the highest expected score.
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/tutorials/05-stochastic-games.md
git commit -m "docs: add 2048 as real-world stochastic game example"
```

---

### Task 5: Concept page — Chance Nodes + 2048

**Files:**
- Modify: `docs/docs/concepts/chance-nodes.md`

- [ ] **Step 1: Read the current page and add 2048 examples**

Read `docs/docs/concepts/chance-nodes.md` in full. Add content at these insertion points:

**In the opening "The problem" section**, add 2048 to the list of examples:

After the existing examples (backgammon, poker, etc.), add:
```markdown
In **2048**, after every slide move, the game places a random tile (2 with 90% probability, 4 with 10%) on a random empty cell. The player must plan without knowing which tile will appear or where.
```

**In the "Open-loop" section**, add a concrete 2048 example:

```markdown
### 2048: A natural open-loop game

2048 is perfectly suited to open-loop MCTS. The player commits to a direction (Up/Down/Left/Right) before learning the tile outcome. Each playout simulates a complete game with different random tile placements, and the statistics naturally average over all possible outcomes.

With open-loop, the tree stores one node per board position (after the slide, before the tile spawn). Different playouts through that node experience different tile spawns, and the node's average reward reflects the expected value across all possibilities.

**[Try it →](/playground)** and watch MCTS evaluate 4 directions against hundreds of random futures.
```

**In the decision guide** (if there is one), add:
```markdown
- **2048**: Open-loop (tile spawn has ~20 outcomes per step; closed-loop would create massive trees)
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/concepts/chance-nodes.md
git commit -m "docs: add 2048 examples to chance nodes concept page"
```

---

### Task 6: Intro page — mention famous games

**Files:**
- Modify: `docs/docs/intro.md`

- [ ] **Step 1: Read the intro page and add game mentions**

Read `docs/docs/intro.md` in full. Add a brief mention of the playground games in the learning path section:

After the existing "Start here" section or site overview, add:

```markdown
### Learn with real games

The tutorials use simple games to teach concepts, but the [Playground](/playground) lets you experience MCTS on games you already know:

- **Tic-Tac-Toe** — Watch MCTS-Solver prove that perfect play is a draw
- **Connect Four** — Challenge MCTS to a deeper strategic game
- **2048** — See how MCTS handles randomness by averaging over possible futures
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/intro.md
git commit -m "docs: mention famous games in intro page learning path"
```

---

## Verification

After all tasks:
```bash
cd docs && npm run build    # docs site builds with all changes
```

All changes are prose/MDX edits — no Rust code, no tests to run. Verify by building the docs site and spot-checking the rendered pages.
