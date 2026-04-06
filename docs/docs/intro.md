---
sidebar_position: 1
---

# MCTS Documentation

This site teaches Monte Carlo Tree Search and how to use the `mcts` Rust crate.

The material serves two audiences: those learning MCTS from scratch, and experienced practitioners who need a production-grade implementation. Everything here applies to both.

## Site structure

- **[Tutorials](/docs/tutorials/01-what-is-mcts)** -- Build working MCTS programs step by step, from theory through two-player games, solvers, and neural network priors.
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
