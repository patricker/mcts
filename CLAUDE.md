# MCTS Crate

> **Wiki**: Run `wiki briefing` for pre-compiled cross-repo context, or `wiki search <term>` to find specific topics. Use the `/wiki` skill for full command reference. Wiki lives at `/home/peter/code/wiki/`.


High-performance, lock-free Monte Carlo Tree Search library for Rust.
Fork of zxqfl/mcts, significantly modernized.

## Project structure

```
Cargo.toml          # Workspace root (members: ".", "mcts-wasm", "mcts-dynamic")
src/                # Core library
  lib.rs            # Public API: MCTS, GameState, Evaluator traits, MCTSManager
  search_tree.rs    # SearchNode, MoveInfo, playout logic, solver/bounds propagation
  tree_policy.rs    # TreePolicy trait, UCTPolicy, AlphaGoPolicy, PolicyRng
  transposition_table.rs  # TranspositionTable trait, ApproxTable
  batch.rs          # Batched neural net evaluation
  atomics.rs        # Atomic re-exports
examples/           # 5 runnable examples
tests/
  mcts_tests.rs     # 111 integration tests
  golden/           # Cross-language golden test definitions (JSON)
benches/bench.rs    # Criterion benchmarks
mcts-dynamic/       # Runtime-polymorphic adapter for language bindings
mcts-wasm/          # WASM bindings crate (wasm-bindgen, cdylib)
docs/               # Docusaurus 3 site (TypeScript)
```

## Commands

- `cargo test` — run all tests (111 core + 26 mcts-dynamic + doc tests)
- `cargo test --test mcts_tests` — core integration tests only
- `cargo test -p mcts-dynamic` — dynamic adapter tests + golden tests
- `cargo clippy` — lint (must stay at 0 warnings)
- `cargo bench` — criterion benchmarks
- `cd mcts-wasm && wasm-pack build --target web` — build WASM package
- `cd docs && npm run build` — build docs site
- `cd docs && npm start` — dev server for docs site

## Key conventions

- All public API items have rustdoc comments
- `TreePolicy::MoveEvaluation` requires `Sync + Send + Default`
- Negamax perspective: decision nodes negate child bounds; chance nodes do NOT negate
- `negate_bound()` is sentinel-safe: `i32::MIN ↔ i32::MAX`, otherwise `-v`
- Score-Bounded MCTS and Solver are independent opt-in features (`score_bounded_enabled()`, `solver_enabled()`)
- Bounds→proven is one-directional (converged bounds set proven value, NOT the reverse)

## Docs site

- Docusaurus 3.9, TypeScript, classic preset
- `remark-code-region` is installed and configured — use `// #region name` / `// #endregion name` tags in Rust source (tests/examples) and embed in MDX via `region` prop on code blocks
- Prism languages: rust, toml
- Pages: home (`/`), docs (`/docs`), playground (`/playground`), API link to docs.rs
- No blog

## Dynamic adapter crate

- `mcts-dynamic/` — workspace member, runtime-polymorphic adapter for language bindings
- Bridges static generics (`MCTSManager<Spec>`) to trait-object-based API (`DynMCTSManager`)
- Key traits: `GameCallbacks` (game state), `EvalCallbacks` (evaluator) — dyn-safe, host languages implement these
- Always uses `AlphaGoPolicy` internally (PUCT); UCT behavior via uniform priors
- Built-in `RandomRollout` evaluator; custom evaluators via `EvalCallbacks`
- Golden tests in `tests/golden/golden_tests.json` — shared across all language bindings
- Dependencies: mcts (path), rand 0.8

## WASM crate

- `mcts-wasm/` — workspace member, `cdylib` + `rlib`
- Dependencies: mcts (path), wasm-bindgen, serde, serde-wasm-bindgen, getrandom (js feature)
- Release profile at workspace root: `opt-level = "s"`, `lto = true`