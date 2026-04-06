# Naive Reader Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the top documentation gaps found by naive reader testing: jargon definitions, performance numbers, evaluator guidance, and cross-language integration mention.

**Architecture:** Small targeted edits to existing MDX files. No new pages except one how-to. No code changes.

**Tech Stack:** MDX (Docusaurus)

---

## File Structure

| File | Change |
|------|--------|
| `docs/docs/tutorials/02-first-search.md` | Define "principal variation" on first use |
| `docs/docs/tutorials/04-solving-games.md` | Separate solver vs score bounds more clearly |
| `docs/docs/intro.md` | Add maturity signals, mention non-Rust paths |
| `docs/docs/how-to/parallel-search.md` | Make time-control pattern more prominent |
| `docs/docs/reference/glossary.md` | Add missing jargon definitions |
| `docs/docs/concepts/exploration-exploitation.md` | Add virtual loss tuning guidance |

---

### Task 1: Jargon definitions on first use

**Files:**
- Modify: `docs/docs/tutorials/02-first-search.md`
- Modify: `docs/docs/reference/glossary.md`
- Modify: `docs/docs/concepts/exploration-exploitation.md`

- [ ] **Step 1: Define "principal variation" on first use**

In `docs/docs/tutorials/02-first-search.md`, find the line:

```
`principal_variation_states(10)` extracts the best sequence of states -- the path the search considers strongest, up to 10 moves deep.
```

Replace with:

```markdown
`principal_variation_states(10)` extracts the **principal variation** (PV) -- the best sequence of moves found by search, following the most-visited child at each level. Think of it as "what MCTS thinks will happen if both sides play optimally." This returns up to 10 states along that path.
```

- [ ] **Step 2: Add virtual loss tuning guidance**

In `docs/docs/concepts/exploration-exploitation.md`, find the section discussing virtual loss. After the explanation of what virtual loss does, add:

```markdown
:::tip Choosing a virtual loss value
Set virtual loss larger than any realistic evaluation your game can produce. If your evaluator returns values in [0, 100], virtual loss of 500 works. If your evaluator returns [-1, 1], virtual loss of 10 works. The exact value doesn't matter much — it just needs to be "obviously bad" so threads avoid piling onto the same path. Virtual loss is only meaningful during parallel search; set it to 0 for single-threaded use.
:::
```

- [ ] **Step 3: Add missing glossary terms**

In `docs/docs/reference/glossary.md`, add these entries in alphabetical order:

```markdown
**Branching factor.**
The average number of legal moves at each game position. Tic-Tac-Toe has branching factor ≤ 9, Chess ~30, Go ~250. Higher branching factors make exhaustive search harder and favor MCTS over minimax.

**Leaf (node).**
A node at the frontier of the search tree that has been evaluated but not yet expanded into children. Each playout extends the tree by one leaf. Not to be confused with a terminal node (which has no legal moves).

**Playout.**
One complete iteration of the MCTS algorithm: selection (walk the tree), expansion (add a leaf), simulation/evaluation (score the leaf), and backpropagation (update statistics along the path). Also called an "iteration" or "rollout" in some literature. Run more playouts for stronger play.
```

- [ ] **Step 4: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 5: Commit**

```bash
git add docs/docs/tutorials/02-first-search.md docs/docs/reference/glossary.md docs/docs/concepts/exploration-exploitation.md
git commit -m "docs: define jargon on first use (PV, virtual loss, branching factor, playout)"
```

---

### Task 2: Intro page — maturity signals + non-Rust mention

**Files:**
- Modify: `docs/docs/intro.md`

- [ ] **Step 1: Add maturity and integration info**

Read `docs/docs/intro.md` in full. After the "Learn with real games" section (recently added), add:

```markdown
### Project status

- **123 integration tests**, all passing, plus golden cross-language tests
- **Zero clippy warnings** — strict Rust linting
- Lock-free parallel search verified on x86-64 with correct Acquire/Release memory ordering
- Benchmarked: ~250k playouts/sec single-threaded on a simple game (CountingGame), ~40k playouts/sec on Mancala (realistic two-player game)
- Available on [GitHub](https://github.com/patricker/mcts)

### Using from other languages

The core library is Rust, but a **runtime-polymorphic adapter** (`mcts-dynamic`) enables language bindings. Games and evaluators are defined via trait objects (`GameCallbacks`, `EvalCallbacks`) using strings for moves — no Rust generics required. Overhead is ~1.4x for realistic games ([benchmarked](https://github.com/patricker/mcts)). WASM bindings power the [Playground](/playground).
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/intro.md
git commit -m "docs: add maturity signals and non-Rust integration mention to intro"
```

---

### Task 3: Tutorial 4 — clearer solver vs bounds separation

**Files:**
- Modify: `docs/docs/tutorials/04-solving-games.md`

- [ ] **Step 1: Add a reader guidance note**

Read `docs/docs/tutorials/04-solving-games.md`. Near the top (after the intro paragraph), add:

```markdown
:::info Two independent features
This tutorial covers two separate features that can be used independently:

1. **MCTS-Solver** (first half) — proves positions as Win/Loss/Draw. Enable with `solver_enabled() = true`. Start here.
2. **Score-Bounded MCTS** (second half) — tracks minimax score bounds. Enable with `score_bounded_enabled() = true`. Skip this section on first read if you don't need exact scores.

Most games only need the solver. Score bounds are useful when you care about *margin of victory*, not just win/loss.
:::
```

- [ ] **Step 2: Add a "skip" marker before score bounds**

Find the heading where Score-Bounded MCTS starts. Add before it:

```markdown
---

*The rest of this tutorial covers Score-Bounded MCTS — an advanced feature for games with numeric scores. If you only need Win/Loss/Draw classification, you can skip ahead to [Tutorial 5: Stochastic Games](./05-stochastic-games.md).*

---
```

- [ ] **Step 3: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 4: Commit**

```bash
git add docs/docs/tutorials/04-solving-games.md
git commit -m "docs: separate solver vs bounds in tutorial 4 with skip guidance"
```

---

### Task 4: Time-control prominence in parallel search how-to

**Files:**
- Modify: `docs/docs/how-to/parallel-search.md`

- [ ] **Step 1: Make time-based search more prominent**

Read `docs/docs/how-to/parallel-search.md` in full. The time-controlled search (`playout_parallel_for`) is already there but buried. Add a prominent callout near the top of the file (after the intro):

```markdown
:::tip Most common pattern: time-controlled search
For games with a clock (e.g., "2 seconds per move"), use time-based search:

```rust
use std::time::Duration;

// Search for 2 seconds on 4 threads
mcts.playout_parallel_for(Duration::from_secs(2), 4);
let best = mcts.best_move();
```

This is what tournament engines use. The search runs as many playouts as it can within the time budget, then stops. See below for more options.
:::
```

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit**

```bash
git add docs/docs/how-to/parallel-search.md
git commit -m "docs: make time-controlled search pattern prominent"
```

---

### Task 5: Embed example output in Nim solver demo

**Files:**
- Modify: `docs/docs/tutorials/04-solving-games.md`

The naive readers noted that code references (`reference="examples/nim_solver.rs#region"`) don't show inline. The existing pattern embeds output files. Let's make sure the nim_solver output is embedded.

- [ ] **Step 1: Check if output file exists and is referenced**

Run: `ls examples/output/nim_solver.txt`

If the output exists but isn't referenced in Tutorial 4, add a reference after the "Running the solver" section:

```markdown
Expected output:

```text reference="examples/output/nim_solver.txt"
```
```

If it's already referenced, skip this task.

- [ ] **Step 2: Verify docs build**

Run: `cd docs && npm run build`

- [ ] **Step 3: Commit (if changes made)**

```bash
git add docs/docs/tutorials/04-solving-games.md
git commit -m "docs: embed nim solver output in tutorial 4"
```

---

## Verification

```bash
cd docs && npm run build    # all changes build cleanly
```

## What This Fixes

| # | Naive Reader Issue | Task |
|---|-------------------|------|
| 1 | No maturity/performance signals | Task 2 (intro) |
| 2 | mcts-dynamic / non-Rust undocumented | Task 2 (intro) |
| 3 | No time-control guidance | Task 4 (parallel how-to) |
| 5 | "Principal variation" undefined on first use | Task 1 |
| 6 | Virtual loss guidance missing | Task 1 |
| 7 | Score bounds too fast in T4 | Task 3 |
| 9 | MCTS jargon undefined | Task 1 (glossary) |

## What This Does NOT Fix (Future Work)

| # | Issue | Reason |
|---|-------|--------|
| 4 | Evaluator-writing how-to | New content page — larger scope, plan separately |
| 8 | Competitor comparison | Marketing/positioning decision, not a docs bug |
| 10 | Full Python bindings tutorial | Depends on mcts-python crate (not yet built) |
