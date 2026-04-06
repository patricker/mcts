---
sidebar_position: 1
id: parallel-search
---

# Run Parallel Search

Run MCTS playouts across multiple CPU threads to search faster.

**You will learn to:**
- Choose between fixed-count, time-limited, and background parallel search
- Configure virtual loss to reduce thread contention

**Prerequisites:** Complete [Your First Search](../tutorials/02-first-search).

:::tip Most common pattern: time-controlled search
For games with a clock (e.g., "2 seconds per move"), use time-based search:

```rust
use std::time::Duration;

// Search for 2 seconds on 4 threads
mcts.playout_parallel_for(Duration::from_secs(2), 4);
let best = mcts.best_move();
```

This is what tournament engines use. The search runs as many playouts as it can within the budget, then stops. See below for details.
:::

## Three parallel APIs

### Fixed playout count

Run exactly `n` playouts distributed across `threads` threads. Blocks until all playouts complete.

```rust
// 10,000 playouts across 4 threads
mcts.playout_n_parallel(10_000, 4);
```

Use this when you want a deterministic amount of search work per move.

### Time-limited

Search for a fixed duration, then stop. Blocks for the full duration.

```rust
use std::time::Duration;

// Search for 1 second on 4 threads
mcts.playout_parallel_for(Duration::from_secs(1), 4);
```

Use this for tournament play with time controls.

### Background (async) search

Start search in the background. Search runs continuously until you stop it.

```rust
// Start searching on 4 threads
let search = mcts.playout_parallel_async(4);

// ... do other work ...

// Stop searching (also stops on drop)
search.halt();
```

The returned `AsyncSearch` holds a mutable borrow on the manager, so you cannot read results until you halt. Use this for pondering during the opponent's turn.

### Owned async search

If you need to move the manager into the search (e.g., across an `await` boundary), use the owned variant:

```rust
let search = mcts.into_playout_parallel_async(4);

// Stops search and returns the MCTSManager
let mcts = search.halt();
let best = mcts.best_move();
```

## Configure virtual loss

When multiple threads descend the tree simultaneously, they tend to follow the same path. Virtual loss temporarily penalizes nodes being visited by other threads, spreading threads across the tree.

Override `virtual_loss()` on your MCTS config:

```rust
impl MCTS for MyMCTS {
    // ... other associated types ...

    fn virtual_loss(&self) -> i64 {
        1000 // larger than any evaluation your game produces
    }
}
```

Set the value larger than the maximum evaluation your `Evaluator` returns. If your evaluations range from -100 to 100, a virtual loss of 1000 ensures no thread revisits a node already being explored.

Without virtual loss (the default of `0`), parallel search still works correctly but threads may redundantly explore the same subtree.

## Expected result

Parallel search produces the same quality of results as single-threaded search, but faster. With 4 threads and proper virtual loss, expect roughly 3-4x throughput improvement.

## See also

- [Parallel MCTS](../concepts/parallel-mcts) -- how lock-free tree parallelism works
- [Configuration reference](../reference/configuration) -- all MCTS trait options
