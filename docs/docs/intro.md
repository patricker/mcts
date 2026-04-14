---
sidebar_position: 1
---

# MCTS Documentation

This site teaches Monte Carlo Tree Search and how to use the `treant` Rust crate.

The material serves two audiences: those learning MCTS from scratch, and experienced practitioners who need a production-grade implementation. Everything here applies to both.

## Site structure

- **[Tutorials](/docs/tutorials/01-what-is-mcts)** -- Build working MCTS programs step by step, from theory through two-player games, solvers, neural network priors, and Gumbel search.
- **[How-To Guides](/docs/how-to/parallel-search)** -- Task-oriented recipes for parallel search, tree reuse, custom policies, WASM integration, and more.
- **[Concepts](/docs/concepts/algorithm)** -- Deep dives into the algorithm, exploration-exploitation tradeoffs, tree policies, solver bounds, and lock-free parallelism.
- **[Reference](/docs/reference/traits)** -- Trait signatures, configuration options, and a glossary of MCTS terminology.

## Start here

Begin with [What is MCTS?](/docs/tutorials/01-what-is-mcts) -- it covers the algorithm in 10 minutes with an interactive demo. No code required.

### Learn with real games

The tutorials use simple games to teach concepts, but the [Playground](/playground) lets you experience MCTS on games you already know:

- **Tic-Tac-Toe** — Watch MCTS-Solver prove that perfect play is a draw
- **Connect Four** — Challenge MCTS to a deeper strategic game
- **2048** — See how MCTS handles randomness by averaging over possible futures

### Project status

- **123+ integration tests**, all passing, plus golden cross-language tests and 39 Gumbel tests
- **Zero clippy warnings** — strict Rust linting
- Lock-free parallel search with correct Acquire/Release memory ordering for ARM
- Benchmarked: ~250k playouts/sec single-threaded on a trivial game, ~40k playouts/sec on Mancala (realistic two-player)
- Available on [GitHub](https://github.com/patricker/treant)

### Using from other languages

The core library is Rust, but a **runtime-polymorphic adapter** (`treant-dynamic`) enables language bindings without Rust generics. Games and evaluators are defined via trait objects (`GameCallbacks`, `EvalCallbacks`) using strings for moves. Overhead is ~1.4x for realistic games ([benchmarked](https://github.com/patricker/treant)). WASM bindings power the [Playground](/playground).

A separate **`treant-gumbel`** crate provides [Gumbel MuZero search](/docs/tutorials/08-gumbel-search) -- Sequential Halving with Gumbel noise for monotonically improving policies. It reuses `GameState` from the core crate, so any game works with both standard MCTS and Gumbel search.
