---
sidebar_position: 3
id: glossary
---

# Glossary

Alphabetical definitions of MCTS terms as used in this library.

---

**Backpropagation.**
The phase where a playout's evaluation is propagated from the leaf back up to the root, updating visit counts and reward sums at every node along the path. In this library, `Evaluator::interpret_evaluation_for_player()` converts the evaluation to each ancestor's perspective during backprop. See [Algorithm](../concepts/algorithm.md).

**Branching factor.**
The average number of legal moves at each game position. Tic-Tac-Toe has branching factor ≤ 9, Chess ~30, Go ~250. Higher branching factors make exhaustive search harder and favor MCTS over minimax.

**Batched evaluation.**
Processing multiple leaf nodes through the evaluator in a single call, amortizing GPU kernel launch overhead. Implemented via `BatchEvaluator` and `BatchedEvaluatorBridge`, which collects pending evaluations from search threads and dispatches them as a batch. See [Batched Evaluation](../how-to/batched-evaluation.md).

**Chance node.**
A node where the next transition is determined by randomness (dice, cards) rather than player choice. In open-loop mode, outcomes are sampled but not stored in the tree. In closed-loop mode (`closed_loop_chance() = true`), each outcome gets its own child. See [Chance Nodes](../concepts/chance-nodes.md).

**Closed-loop.**
A stochastic game mode where chance outcomes are stored as explicit children in the search tree. Each outcome gets its own subtree with independent statistics. More accurate but produces larger trees. Enabled via `MCTS::closed_loop_chance()`. Contrast with open-loop. See [Chance Nodes](../concepts/chance-nodes.md).

**Cycle behaviour.**
The strategy for handling repeated states when transposition tables create graph cycles. Configured via `MCTS::cycle_behaviour()`. Options: `Ignore`, `UseCurrentEvalWhenCycleDetected`, `PanicWhenCycleDetected`, `UseThisEvalWhenCycleDetected`. See [Configuration](./configuration.md).

**Depth limiting.**
Restricting how deep a playout can descend before forcing leaf evaluation. `max_playout_depth()` is a quality knob (evaluate current node as leaf when exceeded). `max_playout_length()` is a safety cap (panics when exceeded). See [Configuration](./configuration.md).

**Dirichlet noise.**
Random noise added to root move priors during self-play training to encourage exploration of non-obvious moves. Configured via `MCTS::dirichlet_noise()` as `(epsilon, alpha)`: the noisy prior is `(1 - epsilon) * prior + epsilon * Dir(alpha)`. Only affects policies with numeric move evaluations (e.g., `AlphaGoPolicy`). See [Tutorial 6](../tutorials/06-neural-network-priors.md).

**Evaluator.**
The component that assigns values to game states and moves. Implements the `Evaluator<Spec>` trait. Produces a `StateEvaluation` for each leaf and per-move `MoveEvaluation` values (e.g., neural network priors). See [Tutorial 2](../tutorials/02-first-search.md).

**Expansion.**
Creating a new tree node for a previously unvisited state. Occurs when a playout reaches a leaf that has been visited `visits_before_expansion()` times. The evaluator is called to produce move evaluations and a state evaluation. See [Algorithm](../concepts/algorithm.md).

**Exploration constant.**
The parameter `C` in UCB1/PUCT that controls the trade-off between exploitation (high-reward moves) and exploration (low-visit moves). Higher values explore more broadly. `UCTPolicy::new(C)` and `AlphaGoPolicy::new(C)` accept this parameter. See [Exploration vs Exploitation](../concepts/exploration-exploitation.md).

**First Play Urgency (FPU).**
The value assigned to unvisited children during selection. Configured via `MCTS::fpu_value()`. Default (`f64::INFINITY`) forces all children to be tried before any revisit. Set to a finite value (e.g., `0.0`) for neural-network-guided search where priors should control exploration order. See [Tutorial 6](../tutorials/06-neural-network-priors.md).

**GameState.**
The trait defining game rules: legal moves, state transitions, current player, and optional features (terminal classification, chance outcomes, progressive widening). Every game implemented with this library starts here. See [Tutorial 2](../tutorials/02-first-search.md) and [Traits Reference](./traits.md).

**Leaf (node).**
A node at the frontier of the search tree that has been evaluated but not yet expanded into children. Each playout extends the tree by one leaf. Not to be confused with a terminal node (which has no legal moves).

**MCTS-Solver.**
An extension that propagates proven game-theoretic values (win/loss/draw) up the tree. When a terminal node is proven, its value flows upward: a parent is proven Win if any child is proven Loss (from the child's perspective), and proven Loss if all children are proven Win. Enabled via `MCTS::solver_enabled()`. See [Solver and Bounds](../concepts/solver-and-bounds.md) and [Tutorial 4](../tutorials/04-solving-games.md).

**Move evaluation.**
A per-move value produced by the evaluator and consumed by the tree policy. For UCT (`UCTPolicy`), this is `()` -- all moves start equal. For PUCT (`AlphaGoPolicy`), this is an `f64` prior probability from a neural network. Type alias: `MoveEvaluation<Spec>`. See [UCT vs PUCT](../concepts/tree-policies.md).

**Negamax.**
A simplification of minimax where scores are negated at each level: a child's score of +5 becomes -5 from the parent's perspective. Used internally for score-bounded backpropagation: parent's lower bound = max(-child.upper) across all children. See [Solver and Bounds](../concepts/solver-and-bounds.md).

**Node.**
A position in the search tree representing a game state. Contains the list of legal moves (`MoveInfo`), visit statistics, state evaluation, and optional solver/bounds data. Represented by `SearchNode<Spec>` internally, accessed via `NodeHandle`. See [Architecture](../concepts/architecture.md).

**Open-loop.**
A stochastic game mode where chance outcomes are sampled during playouts but not stored as separate children in the tree. Multiple outcomes share the same tree node, which blends their statistics. Lower memory than closed-loop but less precise. This is the default behavior when `closed_loop_chance()` returns `false`. Contrast with closed-loop. See [Chance Nodes](../concepts/chance-nodes.md).

**Playout.**
One complete iteration of the MCTS algorithm: selection (walk the tree), expansion (add a leaf), simulation/evaluation (score the leaf), and backpropagation (update statistics along the path). Also called an "iteration" or "rollout" in some literature. Run more playouts for stronger play.

**Principal variation.**
The best sequence of moves found by search, following the most-visited child at each level. Retrieved via `MCTSManager::principal_variation(depth)`. Always uses argmax selection regardless of temperature. See [Tutorial 2](../tutorials/02-first-search.md).

**Progressive widening.**
A technique that limits the number of children expanded at a node based on visit count, gradually considering more moves as the node is visited more. Implemented via `GameState::max_children(visits)`. Useful for large action spaces. See [Progressive Widening](../how-to/progressive-widening.md).

**Proven value.**
A game-theoretic classification of a node as Win, Loss, Draw, or Unknown. Set by MCTS-Solver through bottom-up propagation from terminal nodes. Represented by the `ProvenValue` enum. Proven subtrees are skipped during selection. See [Solver and Bounds](../concepts/solver-and-bounds.md).

**PUCT (Predictor + UCB applied to Trees).**
The tree policy used by AlphaGo and AlphaZero. Selects children using `(Q(a) + C * P(a) * sqrt(N)) / (1 + n(a))`, where `P(a)` is a prior probability. Implemented by `AlphaGoPolicy`. See [UCT vs PUCT](../concepts/tree-policies.md).

**Rollout.**
A synonym for playout. In classic MCTS, rollouts often involve random play to a terminal state. In this library, the evaluator replaces random rollouts with learned evaluation, though the term is sometimes still used. See [Algorithm](../concepts/algorithm.md).

**Score bounds.**
An interval `[lower, upper]` on the minimax value of a node, tracked by Score-Bounded MCTS. Bounds tighten during backpropagation via negamax. When `lower == upper`, the node's exact value is proven. Represented by `ScoreBounds`. Enabled via `MCTS::score_bounded_enabled()`. See [Solver and Bounds](../concepts/solver-and-bounds.md).

**Selection.**
The phase where the tree policy chooses a path from the root to a leaf by picking the most promising child at each internal node. Balances exploration and exploitation. Implemented by `TreePolicy::choose_child()`. See [Algorithm](../concepts/algorithm.md).

**Simulation.**
Synonym for playout. One complete MCTS iteration from root to leaf and back. See playout.

**State evaluation.**
The value assigned to a game state by the evaluator, representing the estimated quality of the position. Produced by `Evaluator::evaluate_new_state()`. Converted to per-player rewards via `interpret_evaluation_for_player()` during backpropagation. Type alias: `StateEvaluation<Spec>`. See [Tutorial 2](../tutorials/02-first-search.md).

**Temperature.**
A parameter controlling the randomness of post-search move selection. At temperature 0 (default), `best_move()` returns the most-visited move (argmax). At temperature 1, selection probability is proportional to visit count. Configured via `MCTS::selection_temperature()`. See [Tutorial 6](../tutorials/06-neural-network-priors.md).

**Transposition table.**
A hash table that maps game states to existing search nodes, allowing the tree to become a directed acyclic graph (or graph with cycles). Implemented via the `TranspositionTable` trait. `ApproxTable` provides a lock-free approximate implementation. Use `()` for no transposition table. See [Architecture](../concepts/architecture.md) and [Tree Reuse](../how-to/tree-reuse.md).

**Tree policy.**
The algorithm that decides which child to explore at each node during selection. Implements the `TreePolicy<Spec>` trait. Built-in policies: `UCTPolicy` (UCB1) and `AlphaGoPolicy` (PUCT). See [UCT vs PUCT](../concepts/tree-policies.md) and [Custom Tree Policy](../how-to/custom-tree-policy.md).

**UCB1 (Upper Confidence Bound 1).**
The formula underlying UCT: `Q(a) + C * sqrt(2 * ln(N) / n(a))`. Provides a principled balance between exploitation (high `Q`) and exploration (low `n`). See [Exploration vs Exploitation](../concepts/exploration-exploitation.md).

**UCT (Upper Confidence bounds applied to Trees).**
The application of UCB1 to tree search, proposed by Kocsis and Szepesvari (2006). The standard MCTS selection policy. Implemented by `UCTPolicy`. See [UCT vs PUCT](../concepts/tree-policies.md).

**Virtual loss.**
A temporary penalty applied to a node's reward sum during descent in parallel search. Discourages multiple threads from exploring the same path simultaneously. Subtracted when a thread descends through a node, added back during backpropagation. Configured via `MCTS::virtual_loss()`. See [Parallel MCTS](../concepts/parallel-mcts.md) and [Parallel Search](../how-to/parallel-search.md).
