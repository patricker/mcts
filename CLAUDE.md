# Treant

> **Wiki**: Run `wiki briefing` for pre-compiled cross-repo context, or `wiki search <term>` to find specific topics. Use the `/wiki` skill for full command reference. Wiki lives at `/home/peter/code/wiki/`.


High-performance, lock-free Monte Carlo Tree Search library for Rust.
Fork of zxqfl/mcts, significantly modernized. Published as `treant` on crates.io.

## Project structure

```
Cargo.toml          # Workspace root (members: ".", "treant-wasm", "treant-dynamic")
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
treant-dynamic/     # Runtime-polymorphic adapter for language bindings
treant-wasm/        # WASM bindings crate (wasm-bindgen, cdylib)
docs/               # Docusaurus 3 site (TypeScript)
```

## Commands

- `cargo test` — run all tests (111 core + 26 treant-dynamic + doc tests)
- `cargo test --test mcts_tests` — core integration tests only
- `cargo test -p treant-dynamic` — dynamic adapter tests + golden tests
- `cargo clippy` — lint (must stay at 0 warnings)
- `cargo bench` — criterion benchmarks
- `cd treant-wasm && wasm-pack build --target web` — build WASM package
- `cd docs && npm run build` — build docs site
- `cd docs && npm start` — dev server for docs site

## Releasing to crates.io

Releases are driven by git tags. The `.github/workflows/release.yml` workflow
triggers on tags matching `<crate>-v<version>`:

1. Bump version in the relevant `Cargo.toml` (core must be bumped before subcrates if they depend on the new version)
2. Commit: `git commit -am "release: <crate> v<version>"`
3. Tag: `git tag <crate>-v<version>` (e.g. `treant-v0.4.1`, `treant-gumbel-v0.1.1`)
4. Push: `git push && git push --tags`

CI will verify the tag matches the Cargo.toml version, run `cargo test -p <crate>`, publish, and create a GitHub Release.

Publish order for multi-crate releases: `treant` first, wait for indexing (~1 min), then `treant-gumbel` / `treant-dynamic` / `treant-wasm` in parallel.

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

- `treant-dynamic/` — workspace member, runtime-polymorphic adapter for language bindings
- Bridges static generics (`MCTSManager<Spec>`) to trait-object-based API (`DynMCTSManager`)
- Key traits: `GameCallbacks` (game state), `EvalCallbacks` (evaluator) — dyn-safe, host languages implement these
- Always uses `AlphaGoPolicy` internally (PUCT); UCT behavior via uniform priors
- Built-in `RandomRollout` evaluator; custom evaluators via `EvalCallbacks`
- Golden tests in `tests/golden/golden_tests.json` — shared across all language bindings
- Dependencies: treant (path), rand 0.8

## WASM crate

- `treant-wasm/` — workspace member, `cdylib` + `rlib`
- Dependencies: treant (path), wasm-bindgen, serde, serde-wasm-bindgen, getrandom (js feature)
- Release profile at workspace root: `opt-level = "s"`, `lto = true`