---
sidebar_position: 5
id: chance-nodes
---

# Open-Loop vs Closed-Loop

Many games involve randomness: dice rolls, card draws, shuffled tiles. After a player acts, nature determines part of the outcome. MCTS must handle this stochastic transition without losing the statistical guarantees that make it work.

The central question is whether to represent random outcomes in the tree. The two approaches -- open-loop and closed-loop -- make different tradeoffs between memory usage, statistical accuracy, and strategic precision.

## The problem

Consider backgammon. A player moves pieces, then rolls dice. The dice outcome determines what moves are available next turn. If MCTS builds a tree node after the player's move, what does that node's value represent?

The difficulty is that the same tree node might be visited under different dice outcomes across different playouts. Playout 1 rolls a 6-6 (strong) and wins. Playout 2 rolls a 1-2 (weak) and loses. The node's average reward mixes outcomes from different dice states. This average is the expected value across all possible rolls, which is exactly what a player needs to evaluate the position before rolling.

But it loses information. If the player could know the dice roll before choosing the move, the strategy might differ. The expected value is correct for pre-roll decisions but wrong for post-roll decisions.

The same issue arises in card games (what card is drawn), tile games (what tile is revealed), and any system with random transitions. The game tree is no longer a tree of player decisions -- it is a tree of decisions interleaved with nature's random choices.

In **2048**, after every slide move, the game places a random tile (2 with 90% probability, 4 with 10%) on a random empty cell. The player must plan without knowing which tile will appear or where.

## Open-loop: sample but do not store

Open-loop is the default mode. During each playout, chance events are sampled (dice are rolled, cards are drawn) and applied to the game state. But the tree does not record which outcome occurred. The tree node stores aggregate statistics across all sampled outcomes.

The mechanics are straightforward. When `GameState::chance_outcomes()` returns `Some(outcomes)` after a move is applied, the library samples an outcome by probability and applies it via `make_move()`. This happens during playout traversal, invisibly to the tree structure. The resulting state (with the sampled outcome incorporated) continues down the tree, but the child nodes do not know which outcome led to them.

Over many playouts, each node's statistics converge to the expected value across all possible chance outcomes. This is exactly what the [multi-armed bandit analysis](./algorithm) guarantees: the average reward converges to the true expected reward regardless of which samples happened to be drawn.

**Memory efficiency.** Open-loop trees are compact. A position with 6 possible dice outcomes and 20 possible responses has 20 child nodes, not 120. For games with large outcome spaces (a 52-card deck has 52 possible draws), the savings are dramatic.

**Convergence.** The law of large numbers guarantees that node statistics converge to the true expected value. The variance introduced by sampling different outcomes on different visits is additional noise on top of the usual MCTS sampling noise, but it averages out at the same rate. Open-loop MCTS retains all the convergence guarantees of standard MCTS.

**When it works.** Open-loop is correct when the player makes decisions before observing the chance outcome, or when the expected value is a sufficient summary. Most board games with dice fit this model: you evaluate a position by averaging over all possible rolls because you cannot condition your strategy on a roll that has not happened yet.

### 2048: A natural open-loop game

2048 is perfectly suited to open-loop MCTS. The player commits to a direction (Up/Down/Left/Right) before learning the tile outcome. Each playout simulates a complete game with different random tile placements, and the statistics naturally average over all possible outcomes.

With open-loop, the tree stores one node per board position (after the slide, before the tile spawn). Different playouts through that node experience different tile spawns, and the node's average reward reflects the expected value across all possibilities.

**[Try it →](/playground)** and watch MCTS evaluate 4 directions against hundreds of random futures.

## Closed-loop: one node per outcome

Closed-loop chance creates separate tree nodes for each random outcome. Enable it with `closed_loop_chance() -> true` and implement `chance_outcomes()` on your `GameState`.

When the search encounters a state with pending chance outcomes, it creates a **chance node** -- a special node whose children represent the possible outcomes rather than player decisions. Each child is selected by probability sampling (not by UCB/PUCT), and each maintains its own statistics.

The key difference: playout 1 might descend through the "rolled 6-6" child and playout 2 through the "rolled 1-2" child. Each child's statistics reflect only that outcome. The parent's value is the probability-weighted combination of its children's values.

**Strategic precision.** Closed-loop trees can represent outcome-dependent strategies. If rolling 6-6 should lead to an aggressive plan and rolling 1-2 to a defensive one, the separate subtrees capture this. Open-loop merges both strategies into a single average, losing the conditional information.

**Memory cost.** Every distinct chance outcome creates a new branch in the tree. A position with 21 possible dice rolls (2d6) multiplies the tree width by 21. In games with continuous or high-cardinality random events, the tree becomes impractically large.

**Selection at chance nodes.** Children of chance nodes are selected by probability sampling, not by UCB/PUCT. This is correct because the player does not choose the outcome -- nature does. More probable outcomes are sampled more often, so their statistics converge faster. Rare outcomes converge slowly, which mirrors their limited impact on expected value.

## Perspective at chance nodes

Chance nodes do not switch the player. The same player acts before and after the random event. A backgammon player chooses a move, dice are rolled, and that same player moves again based on the roll.

This matters for backpropagation. At decision nodes with alternating players, rewards are negated when propagating upward (negamax). At chance nodes, rewards pass through without negation -- the parent and child share the same player's perspective.

The library handles this automatically. Chance nodes are flagged (`is_chance = true`) and backpropagation skips negation for them. The `current_player()` method should return the same player before and after a chance event -- this is how the library knows the perspective does not change.

## Proving through chance

MCTS-Solver extends to chance nodes with conservative rules:

- **All outcomes Win implies parent Win.** If the player wins regardless of what nature does, the position is won.
- **All outcomes Loss implies parent Loss.** If every possible outcome leads to a loss, the position is lost.
- **Mixed outcomes imply Draw.** If some outcomes win and some lose, the conservative classification is Draw. This is strictly correct only for win/loss games; in practice, the expected value might strongly favor one side, but the solver cannot prove it without resolving every outcome.

Score-Bounded MCTS handles chance nodes with weighted averages rather than max/min:

```
parent_lower = floor(sum of prob_i * child_i_lower)
parent_upper = ceil(sum of prob_i * child_i_upper)
```

No negation (same player), and the weighted average replaces the max over children (because nature chooses randomly, not adversarially). The floor/ceil operations ensure the integer bounds are conservative.

The weighted average means bounds tighten slowly for chance nodes. All children must have known bounds before the parent can improve on `[i32::MIN, i32::MAX]`. If even one outcome is unbounded, the parent stays unbounded. This is conservative but correct -- you cannot bound the expected value without bounding every component.

## evaluate_existing_state

Open-loop creates a subtle complication. Different playouts visit the same tree node with different actual game states (because the chance outcomes differed). The evaluator's `evaluate_existing_state` method is called on revisits, receiving the stored evaluation and the current (possibly different) game state.

For simple evaluators that ignore the state and return the stored evaluation, this is a no-op. For evaluators that depend on the actual game state (e.g., a neural network that re-evaluates), the method must handle the fact that the state may differ from the one that created the node.

This is why `evaluate_existing_state` exists as a separate method from `evaluate_new_state`. New-state evaluation happens once, when the node is created. Existing-state evaluation happens on every subsequent visit, potentially with a different state each time.

In closed-loop mode, this issue does not arise. Each chance outcome has its own child node, and revisits to that child always have the same underlying state (the outcome is determined by which child was selected). The `evaluate_existing_state` method is still called, but the state argument matches the one from creation.

For most evaluators, the simplest implementation of `evaluate_existing_state` clones the stored evaluation. This is correct for both open-loop and closed-loop modes. Only implement state-dependent re-evaluation if you have a specific reason to (e.g., an evaluator that depends on hidden information that changes between visits).

## The tradeoff in practice

The choice between open-loop and closed-loop is ultimately about information. Open-loop answers: "What is this position worth before I know the random outcome?" Closed-loop answers: "What is this position worth given each possible random outcome?"

In backgammon, open-loop is natural. You choose a position, then roll. Your strategy before the roll should account for all possible rolls equally. Open-loop's expected value is exactly the right quantity.

In poker, closed-loop is more natural. After a card is revealed, your strategy should depend on which card it was. The expected value across all cards is the wrong basis for post-reveal decisions. You need per-card strategies, which closed-loop provides.

Many games fall between these extremes. A good heuristic: if the player observes the random outcome before their next decision, consider closed-loop. If the random outcome unfolds gradually or is partially hidden, open-loop is typically sufficient.

## When to use which

**Open-loop (default).** Use for most games with randomness. Dice games, card games where you evaluate positions before drawing. Memory efficient, statistically sound, requires no special implementation beyond `chance_outcomes()`.
- **2048**: Open-loop (tile spawn has ~20 outcomes per step; closed-loop would create massive trees)

**Closed-loop (`closed_loop_chance = true`).** Use when per-outcome strategy matters and the outcome space is small. Poker (knowing which card was dealt changes the optimal play), games with a small number of discrete outcomes (coin flips, d6 rolls). Avoid when the outcome space is large or continuous.

**Neither.** If your game has no randomness, neither feature is relevant. `chance_outcomes()` returns `None` by default and the search proceeds as pure MCTS.

**Hybrid approaches.** Some games benefit from a middle ground. You can use open-loop for high-cardinality chance events (card draws from a 52-card deck) and closed-loop for low-cardinality ones (coin flips). The library does not support this directly -- `closed_loop_chance()` is a global flag -- but you can approximate it by collapsing high-cardinality events into a deterministic expected outcome in your `make_move` implementation and only using `chance_outcomes()` for the low-cardinality events.

## Implementation notes

Chance outcomes are checked after every `make_move()` during playouts. If `chance_outcomes()` returns `Some`, an outcome is sampled and applied, then `chance_outcomes()` is checked again. This supports multiple consecutive chance events (e.g., draw a card, then roll a die) without special handling from the caller.

In open-loop mode, chance events are also resolved at the root state before the first selection step. This ensures the playout starts from a concrete state, not one with pending randomness. In closed-loop mode, the root's pending chance events are handled as tree nodes during selection.

The `chance_outcomes()` method must return probabilities that are positive and sum to 1.0. The library samples outcomes via cumulative probability with floating-point tolerance on the last element -- if floating-point rounding causes the cumulative sum to not quite reach 1.0, the last outcome is selected as a fallback. Negative or zero probabilities produce undefined behavior during sampling.

Multiple consecutive chance events are supported. After each `make_move()`, the library calls `chance_outcomes()` again. If it returns `Some`, another outcome is sampled and applied. This loop continues until `chance_outcomes()` returns `None`, indicating the next decision is a player choice. This handles scenarios like "draw a card, then flip a coin" without requiring the game to combine both events into a single outcome distribution.

In closed-loop mode, each link in the chain of consecutive chance events creates its own chance node in the tree. A "draw then flip" sequence produces a chance node for the draw, with children for each card, and each card's child is another chance node for the flip. The tree accurately represents the full stochastic structure, at the cost of additional branching at each level.
