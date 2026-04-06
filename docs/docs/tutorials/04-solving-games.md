---
sidebar_position: 4
id: 04-solving-games
---

# Proving Wins and Losses

Standard MCTS gives statistical estimates: "this move wins 73% of playouts." MCTS-Solver converts those estimates into proofs. Once every child of a node is proven, the node itself is proven. The search stops guessing and starts knowing.

**You will learn to:**
- Implement `terminal_value()` and `terminal_score()` for game outcomes
- Enable MCTS-Solver and Score-Bounded search
- Read and interpret proven values and score bounds after search

**Prerequisites:** [Two-Player Games](./03-two-player-games.md).

## How the solver proves nodes

Proven values propagate up the tree after each playout. The rules follow minimax logic:

- **Any child proven Loss** -- the parent is proven **Win**. The parent's player picks the child where the opponent loses.
- **All children proven Win** -- the parent is proven **Loss**. Every move leads to the opponent winning. No escape.
- **All children proven, at least one Draw** -- the parent is proven **Draw**. The best available outcome is a tie.
- **Terminal node** -- `terminal_value()` returns the outcome directly. This is where proofs originate.

Values are always from the current player's perspective. A child's Loss means the child's mover loses, which means the parent's mover wins.

## Implementing terminal_value

The solver needs to know the outcome at terminal states. Implement `terminal_value()` on your `GameState`:

```rust reference="examples/nim_solver.rs#nim_game"
```

The key method is `terminal_value()`. When no stones remain, the current player (who cannot move) has lost. The function returns `Some(ProvenValue::Loss)`. For non-terminal states, return `None`.

## Enabling the solver

Turn on MCTS-Solver by returning `true` from `solver_enabled()` in your MCTS config:

```rust reference="examples/nim_solver.rs#solver_config"
```

That single method is all it takes. The tree policy, backpropagation, and best-move selection all adapt automatically.

## Running the solver

The solver proves positions as it searches. Call `root_proven_value()` to read the result:

```rust reference="examples/nim_solver.rs#run_solver"
```

`root_proven_value()` returns one of four values:

| Value | Meaning |
|---|---|
| `ProvenValue::Win` | Current player has a forced win |
| `ProvenValue::Loss` | Current player loses with best play |
| `ProvenValue::Draw` | Best achievable outcome is a tie |
| `ProvenValue::Unknown` | Not yet proven (search still running) |

Nim theory says a position is losing if and only if `stones % 3 == 0`. The solver proves this without knowing the theory -- it discovers the game-theoretic truth through search alone.

### Solving Tic-Tac-Toe

Tic-Tac-Toe is the ideal solver showcase. With only ~5,478 distinct game states, MCTS-Solver proves the entire game tree within a few thousand playouts:

- From an empty board: **Proven Draw** (both players can force a draw with optimal play)
- After X plays center: **Proven Draw** (O can always equalize)
- After X plays corner, O plays non-center: **Proven Win for X** (X has a forced win)

**[Try it in the Playground →](/playground)** — play as X and watch MCTS prove each position. The solver badge shows Win/Loss/Draw for the current position, and each move shows its proven value. Notice how MCTS stops searching once a position is fully resolved.

This is exactly what happens in endgame play in production game engines — the solver kicks in when the remaining tree is small enough to prove, saving computation and guaranteeing optimal moves.

## Score-Bounded MCTS

The solver proves win, loss, or draw -- but some games have scores, not just outcomes. How much did you win by? Score-Bounded MCTS answers that.

Sometimes win/loss/draw is not enough. You want the exact score: by how much does the winner win? Score-Bounded MCTS tracks `[lower, upper]` intervals on the minimax value (the optimal score achievable by both players playing perfectly) at each node. These bounds tighten during search as the tree explores more of the game. When `lower == upper`, the exact minimax value is known.

## Terminal scores

Instead of (or in addition to) `terminal_value()`, implement `terminal_score()` to return exact scores at terminal nodes:

```rust reference="examples/score_bounded.rs#score_game"
```

Notice that `terminal_value()` is not implemented here. The library cross-derives proven values from scores automatically:

- Positive score -- `ProvenValue::Win`
- Negative score -- `ProvenValue::Loss`
- Zero -- `ProvenValue::Draw`

You can implement either `terminal_value()`, `terminal_score()`, or both. If you provide both, the library checks consistency in debug builds.

## Configuring score-bounded search

Score-Bounded MCTS requires both `score_bounded_enabled()` and `solver_enabled()` to return `true`:

```rust reference="examples/score_bounded.rs#score_config"
```

The `visits_before_expansion()` override of `0` expands every node immediately. This helps the solver prove small, fully-enumerable trees faster.

## Reading score bounds

After running playouts, read the converged bounds and proven value from the root:

```rust
let bounds = mcts.root_score_bounds();
println!("Score bounds: [{}, {}]", bounds.lower, bounds.upper);
println!("Converged: {}", bounds.is_proven());

let proven = mcts.root_proven_value();
println!("Proven value: {proven:?}");
```

`root_score_bounds()` returns a `ScoreBounds { lower, upper }` struct. When `lower == upper`, the exact minimax value is known, and `is_proven()` returns `true`. For the score game above, the bounds converge to `[6, 6]` -- P1's optimal play through branch C yields a minimax value of 6.

Child-level bounds are also available through `root_child_stats()`, so you can inspect how each move's score interval tightened during search.

:::note Solver in the Playground
Tic-Tac-Toe has solver enabled — positions are proven in real time. Connect Four does not — the tree is too deep for browser-speed proofs, so it relies on heuristic evaluation. This illustrates the practical boundary: solver is powerful but needs a manageable game tree.
:::

## What's next

The solver proves outcomes in deterministic games. [Tutorial 5](./05-stochastic-games) adds randomness with chance nodes -- dice rolls, card draws, and other stochastic transitions.
