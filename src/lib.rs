//! A high-performance, lock-free Monte Carlo Tree Search library.
//!
//! The following example demonstrates basic usage:
//!
//! ```
//! use mcts::{transposition_table::*, tree_policy::*, *};
//!
//! // A really simple game. There's one player and one number. In each move the player can
//! // increase or decrease the number. The player's score is the number.
//! // The game ends when the number reaches 100.
//! //
//! // The best strategy is to increase the number at every step.
//!
//! #[derive(Clone, Debug, PartialEq)]
//! struct CountingGame(i64);
//!
//! #[derive(Clone, Debug, PartialEq)]
//! enum Move {
//!     Add,
//!     Sub,
//! }
//!
//! impl GameState for CountingGame {
//!     type Move = Move;
//!     type Player = ();
//!     type MoveList = Vec<Move>;
//!
//!     fn current_player(&self) -> Self::Player {
//!         ()
//!     }
//!     fn available_moves(&self) -> Vec<Move> {
//!         let x = self.0;
//!         if x == 100 {
//!             vec![]
//!         } else {
//!             vec![Move::Add, Move::Sub]
//!         }
//!     }
//!     fn make_move(&mut self, mov: &Self::Move) {
//!         match *mov {
//!             Move::Add => self.0 += 1,
//!             Move::Sub => self.0 -= 1,
//!         }
//!     }
//! }
//!
//! impl TranspositionHash for CountingGame {
//!     fn hash(&self) -> u64 {
//!         self.0 as u64
//!     }
//! }
//!
//! struct MyEvaluator;
//!
//! impl Evaluator<MyMCTS> for MyEvaluator {
//!     type StateEvaluation = i64;
//!
//!     fn evaluate_new_state(
//!         &self,
//!         state: &CountingGame,
//!         moves: &Vec<Move>,
//!         _: Option<SearchHandle<MyMCTS>>,
//!     ) -> (Vec<()>, i64) {
//!         (vec![(); moves.len()], state.0)
//!     }
//!     fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
//!         *evaln
//!     }
//!     fn evaluate_existing_state(
//!         &self,
//!         _: &CountingGame,
//!         evaln: &i64,
//!         _: SearchHandle<MyMCTS>,
//!     ) -> i64 {
//!         *evaln
//!     }
//! }
//!
//! #[derive(Default)]
//! struct MyMCTS;
//!
//! impl MCTS for MyMCTS {
//!     type State = CountingGame;
//!     type Eval = MyEvaluator;
//!     type NodeData = ();
//!     type ExtraThreadData = ();
//!     type TreePolicy = UCTPolicy;
//!     type TranspositionTable = ApproxTable<Self>;
//!
//!     fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
//!         CycleBehaviour::UseCurrentEvalWhenCycleDetected
//!     }
//! }
//!
//! let game = CountingGame(0);
//! let mut mcts = MCTSManager::new(
//!     game,
//!     MyMCTS,
//!     MyEvaluator,
//!     UCTPolicy::new(0.5),
//!     ApproxTable::new(1024),
//! );
//! mcts.playout_n_parallel(10000, 4); // 10000 playouts, 4 search threads
//! mcts.tree().debug_moves();
//! assert_eq!(mcts.best_move().unwrap(), Move::Add);
//! assert_eq!(mcts.principal_variation(50), vec![Move::Add; 50]);
//! assert_eq!(
//!     mcts.principal_variation_states(5),
//!     vec![
//!         CountingGame(0),
//!         CountingGame(1),
//!         CountingGame(2),
//!         CountingGame(3),
//!         CountingGame(4),
//!         CountingGame(5)
//!     ]
//! );
//! ```

mod atomics;
pub mod batch;
mod search_tree;
pub mod transposition_table;
pub mod tree_policy;

pub use batch::*;
pub use search_tree::*;
use {transposition_table::*, tree_policy::*};

use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::cell::RefCell;

use {
	atomics::*,
	std::{sync::Arc, thread::JoinHandle, time::Duration},
	vec_storage_reuse::VecStorageForReuse,
};

/// Configuration trait for MCTS search. Defines the game, evaluator,
/// tree policy, and optional features.
pub trait MCTS: Sized + Send + Sync + 'static {
	type State: GameState + Send + Sync + 'static;
	type Eval: Evaluator<Self> + Send + 'static;
	type TreePolicy: TreePolicy<Self> + Send + 'static;
	type NodeData: Default + Sync + Send + 'static;
	type TranspositionTable: TranspositionTable<Self> + Send + 'static;
	type ExtraThreadData: 'static;

	/// Virtual loss for parallel search. Subtracted during descent, added back during backprop.
	fn virtual_loss(&self) -> i64 {
		0
	}
	/// Default value for unvisited children during selection.
	/// `f64::INFINITY` (default) forces all children to be tried before any revisit.
	/// Set to a finite value (e.g. `0.0`) for neural-network-guided search where
	/// the prior should control which children are explored first.
	fn fpu_value(&self) -> f64 {
		f64::INFINITY
	}
	/// Minimum visits to a leaf before expanding it into a tree node.
	fn visits_before_expansion(&self) -> u64 {
		1
	}
	/// Maximum number of tree nodes. Search stops when reached.
	fn node_limit(&self) -> usize {
		usize::MAX
	}
	/// Select the best child after search completes. Override for custom post-search selection.
	fn select_child_after_search<'a>(&self, children: &'a [MoveInfo<Self>]) -> &'a MoveInfo<Self> {
		if self.solver_enabled() {
			// Prefer proven-win children (child's Loss = parent's win)
			if let Some(winner) = children.iter().find(|c| c.child_proven_value() == ProvenValue::Loss) {
				return winner;
			}
			// Prefer proven-draw over proven-loss
			if let Some(drawer) = children.iter().find(|c| c.child_proven_value() == ProvenValue::Draw) {
				return drawer;
			}
		}
		if self.score_bounded_enabled() {
			// Pick the child with the best guaranteed score from parent's perspective.
			// Parent's lower from child = negate(child.upper).
			let best_lower = children
				.iter()
				.map(|c| negate_bound(c.child_score_bounds().upper))
				.max()
				.unwrap_or(i32::MIN);
			if best_lower > i32::MIN {
				return children
					.iter()
					.max_by_key(|c| negate_bound(c.child_score_bounds().upper))
					.unwrap();
			}
		}
		children.iter().max_by_key(|child| child.visits()).unwrap()
	}
	/// `playout` panics when this length is exceeded. Defaults to one million.
	fn max_playout_length(&self) -> usize {
		1_000_000
	}
	/// Maximum depth per playout before forcing leaf evaluation.
	/// Unlike `max_playout_length` (a safety cap), this is a quality knob:
	/// when exceeded, the current node is evaluated as a leaf.
	fn max_playout_depth(&self) -> usize {
		usize::MAX
	}
	/// Optional RNG seed for deterministic search. When set, each thread gets
	/// a reproducible RNG seeded from `seed + thread_id`.
	fn rng_seed(&self) -> Option<u64> {
		None
	}
	/// Dirichlet noise for root exploration during self-play.
	/// Returns `Some((epsilon, alpha))` where noisy prior =
	/// `(1 - epsilon) * prior + epsilon * Dir(alpha)`.
	/// Typical: eps=0.25, alpha=0.03 (Go), alpha=0.3 (Chess).
	/// Only applies when TreePolicy::MoveEvaluation supports noise (e.g. f64).
	fn dirichlet_noise(&self) -> Option<(f64, f64)> {
		None
	}
	/// Temperature for post-search move selection in `best_move()`.
	/// 0.0 (default) = argmax by visits. 1.0 = proportional to visits.
	/// `principal_variation()` always uses argmax regardless of temperature.
	fn selection_temperature(&self) -> f64 {
		0.0
	}
	/// Enable MCTS-Solver: proven game-theoretic values (win/loss/draw)
	/// propagate up the tree, and solved subtrees are skipped during selection.
	/// Requires `GameState::terminal_value()` to classify terminal states.
	/// Default: false (no solver overhead).
	fn solver_enabled(&self) -> bool {
		false
	}
	/// Enable Score-Bounded MCTS: each node tracks `[lower, upper]` bounds
	/// on its minimax value (from the current player's perspective).
	/// Bounds tighten during backpropagation using negamax. When bounds
	/// converge (`lower == upper`), the node's exact value is proven.
	/// Requires `GameState::terminal_score()` to set leaf bounds.
	/// Independent of `solver_enabled()` — both can be active simultaneously.
	/// Default: false.
	fn score_bounded_enabled(&self) -> bool {
		false
	}
	/// Enable closed-loop chance nodes: each chance outcome gets its own
	/// child in the tree, selected by probability sampling. More accurate
	/// per-outcome statistics than open-loop, but larger trees.
	/// Requires discrete, enumerable outcomes via `GameState::chance_outcomes()`.
	/// Default: false (open-loop: outcomes sampled but not stored in tree).
	fn closed_loop_chance(&self) -> bool {
		false
	}
	/// Called during backpropagation for each node on the playout path.
	fn on_backpropagation(&self, _evaln: &StateEvaluation<Self>, _handle: SearchHandle<Self>) {}
	/// How to handle cycles caused by transposition tables.
	fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
		if std::mem::size_of::<Self::TranspositionTable>() == 0 {
			CycleBehaviour::Ignore
		} else {
			CycleBehaviour::PanicWhenCycleDetected
		}
	}
}

/// Thread-local data passed to tree policy and user code during search.
pub struct ThreadData<Spec: MCTS> {
	pub policy_data: TreePolicyThreadData<Spec>,
	pub extra_data: Spec::ExtraThreadData,
}

impl<Spec: MCTS> Default for ThreadData<Spec>
where
	TreePolicyThreadData<Spec>: Default,
	Spec::ExtraThreadData: Default,
{
	fn default() -> Self {
		Self {
			policy_data: Default::default(),
			extra_data: Default::default(),
		}
	}
}

/// Contains the regular thread data + some `Vec`s that we want to reuse the allocation of
/// within `playout`
pub struct ThreadDataFull<Spec: MCTS> {
	tld: ThreadData<Spec>,
	// Storage reuse - as an alternative to SmallVec
	path: VecStorageForReuse<*const MoveInfo<Spec>>,
	node_path: VecStorageForReuse<*const SearchNode<Spec>>,
	players: VecStorageForReuse<Player<Spec>>,
	chance_rng: SmallRng,
}

impl<Spec: MCTS> Default for ThreadDataFull<Spec>
where
	ThreadData<Spec>: Default,
{
	fn default() -> Self {
		Self {
			tld: Default::default(),
			path: VecStorageForReuse::default(),
			node_path: VecStorageForReuse::default(),
			players: VecStorageForReuse::default(),
			chance_rng: SmallRng::from_rng(rand::thread_rng()).unwrap(),
		}
	}
}


/// Per-move evaluation from the tree policy (e.g., `()` for UCT, `f64` prior for AlphaGo).
pub type MoveEvaluation<Spec> = <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::MoveEvaluation;
/// State evaluation produced by the `Evaluator`.
pub type StateEvaluation<Spec> = <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation;
/// The move type for the game state.
pub type Move<Spec> = <<Spec as MCTS>::State as GameState>::Move;
/// The move list type returned by `GameState::available_moves()`.
pub type MoveList<Spec> = <<Spec as MCTS>::State as GameState>::MoveList;
/// The player type for the game state.
pub type Player<Spec> = <<Spec as MCTS>::State as GameState>::Player;
/// Thread-local data for the tree policy.
pub type TreePolicyThreadData<Spec> = <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::ThreadLocalData;

/// Game-theoretic proven value for MCTS-Solver.
/// Stored from the perspective of the player who moved to reach this node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ProvenValue {
	Unknown = 0,
	Win = 1,
	Loss = 2,
	Draw = 3,
}

impl ProvenValue {
	/// Convert from raw u8 representation. Unknown for unrecognized values.
	pub fn from_u8(v: u8) -> Self {
		match v {
			1 => ProvenValue::Win,
			2 => ProvenValue::Loss,
			3 => ProvenValue::Draw,
			_ => ProvenValue::Unknown,
		}
	}
}

/// Proven score interval for Score-Bounded MCTS.
/// Tracks `[lower, upper]` bounds on the true minimax value from the
/// current player's perspective. When `lower == upper`, the value is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreBounds {
	pub lower: i32,
	pub upper: i32,
}

impl ScoreBounds {
	/// No bounds known: `[i32::MIN, i32::MAX]`.
	pub const UNBOUNDED: Self = Self {
		lower: i32::MIN,
		upper: i32::MAX,
	};

	/// Exact proven value: `lower == upper == v`.
	pub fn exact(v: i32) -> Self {
		Self { lower: v, upper: v }
	}

	/// Returns `true` when bounds have converged (`lower == upper`).
	pub fn is_proven(&self) -> bool {
		self.lower == self.upper
	}
}

/// Negate a score bound, mapping sentinels correctly.
/// `i32::MIN` (unbounded below) becomes `i32::MAX` (unbounded above) and vice versa.
pub(crate) fn negate_bound(v: i32) -> i32 {
	match v {
		i32::MIN => i32::MAX,
		i32::MAX => i32::MIN,
		_ => -v,
	}
}

/// Defines the game rules: available moves, state transitions, and players.
pub trait GameState: Clone {
	type Move: Sync + Send + Clone;
	type Player: Sync;
	type MoveList: std::iter::IntoIterator<Item = Self::Move>;

	/// The player whose turn it is.
	fn current_player(&self) -> Self::Player;
	/// Legal moves from this state. Empty means terminal.
	fn available_moves(&self) -> Self::MoveList;
	/// Apply a move, mutating the state in place.
	fn make_move(&mut self, mov: &Self::Move);

	/// Maximum children to expand at this node given the current visit count.
	/// Override for progressive widening, e.g. `(visits as f64).sqrt() as usize`.
	/// Moves are expanded in the order returned by `available_moves()`, so return
	/// them in priority order when using progressive widening.
	fn max_children(&self, _visits: u64) -> usize {
		usize::MAX
	}

	/// When the state is terminal (no available moves), classify the outcome.
	/// Returns the proven value from the perspective of the current player
	/// (the player who would move next, but cannot because the game is over).
	/// If the current player has lost, return `Some(ProvenValue::Loss)`.
	/// Default: `None` (solver treats terminal nodes as Unknown).
	fn terminal_value(&self) -> Option<ProvenValue> {
		None
	}

	/// When the state is terminal, return its exact minimax score from the
	/// current player's perspective. Used by Score-Bounded MCTS to set
	/// exact bounds on terminal nodes.
	/// Default: `None` (score bounds are not set on terminals).
	fn terminal_score(&self) -> Option<i32> {
		None
	}

	/// If the current state requires a chance event (dice roll, card draw)
	/// before the next player decision, return the possible outcomes with
	/// their probabilities. Outcomes are applied via `make_move()`.
	///
	/// Probabilities must be positive and sum to 1.0.
	/// Return `None` for deterministic transitions (the default).
	///
	/// This is called after each `make_move()` during playouts. If the
	/// result is `Some`, an outcome is sampled and applied, then
	/// `chance_outcomes()` is checked again (supporting multiple
	/// consecutive chance events).
	fn chance_outcomes(&self) -> Option<Vec<(Self::Move, f64)>> {
		None
	}
}

/// Evaluates game states for the search. Produces state evaluations
/// and per-move evaluations.
pub trait Evaluator<Spec: MCTS>: Sync {
	type StateEvaluation: Sync + Send;

	/// Evaluate a newly expanded state. Returns per-move evaluations and a state evaluation.
	fn evaluate_new_state(
		&self,
		state: &Spec::State,
		moves: &MoveList<Spec>,
		handle: Option<SearchHandle<Spec>>,
	) -> (Vec<MoveEvaluation<Spec>>, Self::StateEvaluation);

	/// Re-evaluate a previously seen state (e.g., for open-loop chance nodes).
	fn evaluate_existing_state(
		&self,
		state: &Spec::State,
		existing_evaln: &Self::StateEvaluation,
		handle: SearchHandle<Spec>,
	) -> Self::StateEvaluation;

	/// Convert a state evaluation to a reward from the given player's perspective.
	fn interpret_evaluation_for_player(&self, evaluation: &Self::StateEvaluation, player: &Player<Spec>) -> i64;
}

/// Main entry point for running MCTS search. Owns the search tree and provides
/// methods for running playouts and extracting results.
pub struct MCTSManager<Spec: MCTS> {
	search_tree: Arc<SearchTree<Spec>>,
	// thread local data when we have no asynchronous workers
	single_threaded_tld: Option<ThreadDataFull<Spec>>,
	print_on_playout_error: bool,
	selection_rng: RefCell<SmallRng>,
}

impl<Spec: MCTS> MCTSManager<Spec>
where
	ThreadData<Spec>: Default,
{
	/// Create a new search manager with the given game state, config, evaluator,
	/// tree policy, and transposition table.
	pub fn new(
		state: Spec::State,
		manager: Spec,
		eval: Spec::Eval,
		tree_policy: Spec::TreePolicy,
		table: Spec::TranspositionTable,
	) -> Self {
		let selection_rng = match manager.rng_seed() {
			Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(u64::MAX / 2)),
			None => SmallRng::from_rng(rand::thread_rng()).unwrap(),
		};
		let search_tree = Arc::new(SearchTree::new(state, manager, tree_policy, eval, table));
		let single_threaded_tld = None;
		Self {
			search_tree,
			single_threaded_tld,
			print_on_playout_error: true,
			selection_rng: RefCell::new(selection_rng),
		}
	}

	pub fn print_on_playout_error(&mut self, v: bool) -> &mut Self {
		self.print_on_playout_error = v;
		self
	}

	/// Run a single playout (single-threaded).
	pub fn playout(&mut self) {
		// Avoid overhead of thread creation
		if self.single_threaded_tld.is_none() {
			self.single_threaded_tld = Some(self.search_tree.make_thread_data());
		}
		self.search_tree.playout(self.single_threaded_tld.as_mut().unwrap());
	}
	pub fn playout_until<Predicate: FnMut() -> bool>(&mut self, mut pred: Predicate) {
		while !pred() {
			self.playout();
		}
	}
	/// Run `n` playouts sequentially.
	pub fn playout_n(&mut self, n: u64) {
		for _ in 0..n {
			self.playout();
		}
	}
	/// Start asynchronous parallel search. Returns a handle that stops search on drop.
	pub fn playout_parallel_async<'a>(&'a mut self, num_threads: usize) -> AsyncSearch<'a, Spec> {
		assert!(num_threads != 0);
		let stop_signal = Arc::new(AtomicBool::new(false));
		let threads = (0..num_threads)
			.map(|_| {
				spawn_search_thread(
					Arc::clone(&self.search_tree),
					Arc::clone(&stop_signal),
					self.print_on_playout_error,
				)
			})
			.collect();
		AsyncSearch {
			manager: self,
			stop_signal,
			threads,
		}
	}
	/// Like `playout_parallel_async`, but takes ownership of the manager.
	pub fn into_playout_parallel_async(self, num_threads: usize) -> AsyncSearchOwned<Spec> {
		assert!(num_threads != 0);
		let self_box = Box::new(self);
		let stop_signal = Arc::new(AtomicBool::new(false));
		let threads = (0..num_threads)
			.map(|_| {
				spawn_search_thread(
					Arc::clone(&self_box.search_tree),
					Arc::clone(&stop_signal),
					self_box.print_on_playout_error,
				)
			})
			.collect();
		AsyncSearchOwned {
			manager: Some(self_box),
			stop_signal,
			threads,
		}
	}
	/// Run parallel search for the given duration using scoped threads.
	pub fn playout_parallel_for(&mut self, duration: Duration, num_threads: usize) {
		assert!(num_threads != 0);
		let stop_signal = AtomicBool::new(false);
		let search_tree = &*self.search_tree;
		let print_on_playout_error = self.print_on_playout_error;
		std::thread::scope(|s| {
			for _ in 0..num_threads {
				s.spawn(|| {
					let mut tld = search_tree.make_thread_data();
					loop {
						if stop_signal.load(Ordering::SeqCst) {
							break;
						}
						if !search_tree.playout(&mut tld) {
							if print_on_playout_error {
								eprintln!(
									"Node limit of {} reached. Halting search.",
									search_tree.spec().node_limit()
								);
							}
							break;
						}
					}
				});
			}
			std::thread::sleep(duration);
			stop_signal.store(true, Ordering::SeqCst);
		});
	}
	/// Run `n` playouts across multiple threads using scoped threads.
	pub fn playout_n_parallel(&mut self, n: u32, num_threads: usize) {
		if n == 0 {
			return;
		}
		assert!(num_threads != 0);
		let counter = AtomicIsize::new(n as isize);
		let search_tree = &*self.search_tree;
		std::thread::scope(|s| {
			for _ in 0..num_threads {
				s.spawn(|| {
					let mut tld = search_tree.make_thread_data();
					loop {
						let count = counter.fetch_sub(1, Ordering::SeqCst);
						if count <= 0 {
							break;
						}
						search_tree.playout(&mut tld);
					}
				});
			}
		});
	}
	/// The principal variation with full move info handles.
	pub fn principal_variation_info(&self, num_moves: usize) -> Vec<MoveInfoHandle<'_, Spec>> {
		self.search_tree.principal_variation(num_moves)
	}
	/// The best sequence of moves found by search.
	pub fn principal_variation(&self, num_moves: usize) -> Vec<Move<Spec>> {
		self.search_tree
			.principal_variation(num_moves)
			.into_iter()
			.map(|x| x.get_move().clone())
			.collect()
	}
	/// The principal variation as a sequence of game states.
	pub fn principal_variation_states(&self, num_moves: usize) -> Vec<Spec::State> {
		let moves = self.principal_variation(num_moves);
		let mut states = vec![self.search_tree.root_state().clone()];
		for mov in moves {
			let mut state = states[states.len() - 1].clone();
			state.make_move(&mov);
			states.push(state);
		}
		states
	}
	/// Access the underlying search tree.
	pub fn tree(&self) -> &SearchTree<Spec> {
		&self.search_tree
	}
	/// Returns the proven value of the root node (for MCTS-Solver).
	pub fn root_proven_value(&self) -> ProvenValue {
		self.search_tree.root_proven_value()
	}
	/// Returns the score bounds of the root node (for Score-Bounded MCTS).
	pub fn root_score_bounds(&self) -> ScoreBounds {
		self.search_tree.root_score_bounds()
	}
	/// The best move found. Uses temperature-based selection if configured.
	pub fn best_move(&self) -> Option<Move<Spec>> {
		let temperature = self.search_tree.spec().selection_temperature();
		if temperature < 1e-8 {
			self.principal_variation(1).first().cloned()
		} else {
			self.select_move_by_temperature(temperature)
		}
	}

	fn select_move_by_temperature(&self, temperature: f64) -> Option<Move<Spec>> {
		let inv_temp = 1.0 / temperature;
		let weighted: Vec<_> = self
			.search_tree
			.root_node()
			.moves()
			.filter(|c| c.visits() > 0)
			.map(|c| (c.get_move().clone(), (c.visits() as f64).powf(inv_temp)))
			.collect();
		if weighted.is_empty() {
			return None;
		}
		let total: f64 = weighted.iter().map(|(_, w)| w).sum();
		let mut roll: f64 = self.selection_rng.borrow_mut().gen::<f64>() * total;
		for (mov, weight) in &weighted {
			roll -= weight;
			if roll <= 0.0 {
				return Some(mov.clone());
			}
		}
		Some(weighted.last().unwrap().0.clone())
	}
	/// Run a 10-second performance benchmark, calling `f` with nodes/sec each second.
	pub fn perf_test<F>(&mut self, num_threads: usize, mut f: F)
	where
		F: FnMut(usize),
	{
		let search = self.playout_parallel_async(num_threads);
		for _ in 0..10 {
			let n1 = search.manager.search_tree.num_nodes();
			std::thread::sleep(Duration::from_secs(1));
			let n2 = search.manager.search_tree.num_nodes();
			let diff = n2.saturating_sub(n1);
			f(diff);
		}
	}
	pub fn perf_test_to_stderr(&mut self, num_threads: usize) {
		self.perf_test(num_threads, |x| eprintln!("{} nodes/sec", thousands_separate(x)));
	}
	/// Reset the search tree, keeping the same game state and configuration.
	pub fn reset(self) -> Self {
		let search_tree = Arc::try_unwrap(self.search_tree)
			.unwrap_or_else(|_| panic!("Cannot reset while async search is running"));
		let selection_rng = match search_tree.spec().rng_seed() {
			Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(u64::MAX / 2)),
			None => SmallRng::from_rng(rand::thread_rng()).unwrap(),
		};
		Self {
			search_tree: Arc::new(search_tree.reset()),
			print_on_playout_error: self.print_on_playout_error,
			single_threaded_tld: None,
			selection_rng: RefCell::new(selection_rng),
		}
	}
}

impl<Spec: MCTS> MCTSManager<Spec>
where
	Move<Spec>: PartialEq,
	ThreadData<Spec>: Default,
{
	/// Commit to a move: advance the root and preserve the subtree below it.
	/// Returns `Err` if the move is not found, not expanded, or not owned.
	/// Panics if an async search is still running.
	pub fn advance(&mut self, mov: &Move<Spec>) -> Result<(), AdvanceError> {
		let tree = Arc::get_mut(&mut self.search_tree)
			.expect("Cannot advance while async search is running");
		tree.advance_root(mov)?;
		self.single_threaded_tld = None;
		Ok(())
	}
}

impl<Spec: MCTS> MCTSManager<Spec>
where
	MoveEvaluation<Spec>: Clone,
{
	/// Visit counts and average rewards for all root children.
	pub fn root_child_stats(&self) -> Vec<ChildStats<Spec>> {
		self.search_tree.root_child_stats()
	}
}

// https://stackoverflow.com/questions/26998485/rust-print-format-number-with-thousand-separator
fn thousands_separate(x: usize) -> String {
	let s = format!("{}", x);
	let chunks: Vec<&str> = s
		.as_bytes()
		.rchunks(3)
		.rev()
		.map(|chunk| std::str::from_utf8(chunk).unwrap())
		.collect();
	chunks.join(",")
}

/// Handle for an in-progress asynchronous search. Stops search on drop.
#[must_use]
pub struct AsyncSearch<'a, Spec: 'a + MCTS> {
	manager: &'a mut MCTSManager<Spec>,
	stop_signal: Arc<AtomicBool>,
	threads: Vec<JoinHandle<()>>,
}

impl<'a, Spec: MCTS> AsyncSearch<'a, Spec> {
	pub fn halt(self) {}
	pub fn num_threads(&self) -> usize {
		self.threads.len()
	}
}

impl<'a, Spec: MCTS> Drop for AsyncSearch<'a, Spec> {
	fn drop(&mut self) {
		self.stop_signal.store(true, Ordering::SeqCst);
		drain_join_unwrap(&mut self.threads);
	}
}

/// Owned variant of `AsyncSearch`. Call `halt()` to stop and recover the manager.
#[must_use]
pub struct AsyncSearchOwned<Spec: MCTS> {
	manager: Option<Box<MCTSManager<Spec>>>,
	stop_signal: Arc<AtomicBool>,
	threads: Vec<JoinHandle<()>>,
}

impl<Spec: MCTS> AsyncSearchOwned<Spec> {
	fn stop_threads(&mut self) {
		self.stop_signal.store(true, Ordering::SeqCst);
		drain_join_unwrap(&mut self.threads);
	}
	pub fn halt(mut self) -> MCTSManager<Spec> {
		self.stop_threads();
		*self.manager.take().unwrap()
	}
	pub fn num_threads(&self) -> usize {
		self.threads.len()
	}
}

impl<Spec: MCTS> Drop for AsyncSearchOwned<Spec> {
	fn drop(&mut self) {
		self.stop_threads();
	}
}

impl<Spec: MCTS> From<MCTSManager<Spec>> for AsyncSearchOwned<Spec> {
	/// An `MCTSManager` is an `AsyncSearchOwned` with zero threads searching.
	fn from(m: MCTSManager<Spec>) -> Self {
		Self {
			manager: Some(Box::new(m)),
			stop_signal: Arc::new(AtomicBool::new(false)),
			threads: Vec::new(),
		}
	}
}

fn spawn_search_thread<Spec: MCTS>(
	search_tree: Arc<SearchTree<Spec>>,
	stop_signal: Arc<AtomicBool>,
	print_on_playout_error: bool,
) -> JoinHandle<()>
where
	ThreadData<Spec>: Default,
{
	std::thread::spawn(move || {
		let mut tld = search_tree.make_thread_data();
		loop {
			if stop_signal.load(Ordering::SeqCst) {
				break;
			}
			if !search_tree.playout(&mut tld) {
				if print_on_playout_error {
					eprintln!(
						"Node limit of {} reached. Halting search.",
						search_tree.spec().node_limit()
					);
				}
				break;
			}
		}
	})
}

fn drain_join_unwrap(threads: &mut Vec<JoinHandle<()>>) {
	let join_results: Vec<_> = threads.drain(..).map(|x| x.join()).collect();
	for x in join_results {
		x.unwrap();
	}
}

/// Strategy for handling graph cycles caused by transposition tables.
pub enum CycleBehaviour<Spec: MCTS> {
	/// Ignore cycles (may cause infinite loops without depth limits).
	Ignore,
	/// Break the cycle and evaluate the current state.
	UseCurrentEvalWhenCycleDetected,
	/// Panic on cycle detection (useful for debugging).
	PanicWhenCycleDetected,
	/// Break the cycle and use this specific evaluation.
	UseThisEvalWhenCycleDetected(StateEvaluation<Spec>),
}
