---
sidebar_position: 2
id: 02-first-search
---

# Your First Search

You'll build a complete MCTS search from scratch. The game: increment or decrement a counter toward a target of 100. Two moves per turn, one obvious best choice. Simple enough to verify, complex enough to demonstrate every part of the API.

**You will learn to:**
- Implement `GameState` for a single-player game
- Implement `Evaluator` with state evaluation and player-perspective reward
- Configure and run MCTS search, and interpret the results

**Prerequisites:** [What is MCTS?](./01-what-is-mcts.md) and a Rust toolchain.

## Add the dependency

```toml
[dependencies]
mcts = "0.3"
```

## Define the game

Every MCTS game implements the `GameState` trait: what moves are available, whose turn it is, and how moves change the state.

```rust reference="examples/counting_game.rs#game_definition"
```

`CountingGame` wraps a single `i64` counter. Each turn, the player can `Add` (increment) or `Sub` (decrement). When the counter reaches 100, `available_moves()` returns an empty vec -- the game is over.

`current_player()` returns `()` because this is a single-player optimization problem. There is no opponent.

`make_move()` mutates the state in place. MCTS clones the state internally before calling this, so the original is preserved.

## Evaluate positions

The `Evaluator` trait tells MCTS how good a position is and how to score moves.

```rust reference="examples/counting_game.rs#evaluator"
```

Three methods:

- **`evaluate_new_state()`** returns a pair: per-move evaluations (here `()` for each move, meaning no prior bias) and a state evaluation (the counter value itself). Higher counter = closer to 100 = better.
- **`interpret_evaluation_for_player()`** converts the state evaluation into a reward for the given player. With a single player, this just returns the evaluation directly.
- **`evaluate_existing_state()`** handles re-evaluation when the search revisits a node (through different move paths that reach the same game state — called transpositions). Here, it returns the same value.

## Configure the search

The `MCTS` trait wires the game, evaluator, tree policy, and transposition table together into a single configuration type.

```rust reference="examples/counting_game.rs#mcts_config"
```

`MyMCTS` is a zero-sized type that exists solely to connect the associated types.

- **`TreePolicy = UCTPolicy`** -- the tree policy (the algorithm that decides which child to explore next — here, UCT from Tutorial 1).
- **`TranspositionTable = ApproxTable`** -- detect when different move sequences reach the same counter value.
- **`virtual_loss()`** returns 500. During parallel search (covered in the [how-to guide](/docs/how-to/parallel-search)), a thread temporarily penalizes a node it's exploring so other threads avoid duplicating work. The value should be larger than any realistic evaluation.
- **`cycle_behaviour()`** tells the search what to do when different move paths loop back to the same position. `UseCurrentEvalWhenCycleDetected` stops expanding and uses the node's current evaluation.

## Run it

```rust reference="examples/counting_game.rs#run_search"
```

`MCTSManager::new()` takes five arguments: the initial game state, the MCTS configuration, the evaluator, the tree policy (with exploration constant `C=5.0`), and the transposition table (1024-slot approximate table).

`playout_n(100_000)` runs 100,000 iterations. Each iteration walks the four phases: select, expand, simulate, backpropagate.

`principal_variation_states(10)` extracts the **principal variation** (PV) -- the best sequence of moves found by search, following the most-visited child at each level. Think of it as "what MCTS thinks will happen if both sides play optimally." This returns up to 10 states along that path.

`debug_moves()` prints statistics for each child of the root: visit count, average reward, and move name. You'll see `Add` with far more visits and a higher average reward than `Sub`.

Expected output (exact numbers vary between runs):

```text reference="examples/output/counting_game.txt"
```

The search overwhelmingly prefers `Add` — nearly all 100,000 playouts go through it. The principal variation climbs 0, 1, 2, 3, ... toward 100. MCTS found the obvious optimal strategy through sampling alone.

## The full picture

The complete example is at `examples/counting_game.rs` in the repository. Run it:

```bash
cargo run --example counting_game
```

## What's next

The counting game has one player and one obvious strategy. In [Two-Player Games](./03-two-player-games.md), you'll add an opponent and adversarial evaluation.
