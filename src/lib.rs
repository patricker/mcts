//! This is a library for Monte Carlo tree search.
//!
//! It is still under development and the documentation isn't good. However, the following example may be helpful:
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

pub trait MCTS: Sized + Send + Sync + 'static {
	type State: GameState + Send + Sync + 'static;
	type Eval: Evaluator<Self> + Send + 'static;
	type TreePolicy: TreePolicy<Self> + Send + 'static;
	type NodeData: Default + Sync + Send + 'static;
	type TranspositionTable: TranspositionTable<Self> + Send + 'static;
	type ExtraThreadData: 'static;

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
	fn visits_before_expansion(&self) -> u64 {
		1
	}
	fn node_limit(&self) -> usize {
		usize::MAX
	}
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
	fn on_backpropagation(&self, _evaln: &StateEvaluation<Self>, _handle: SearchHandle<Self>) {}
	fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
		if std::mem::size_of::<Self::TranspositionTable>() == 0 {
			CycleBehaviour::Ignore
		} else {
			CycleBehaviour::PanicWhenCycleDetected
		}
	}
}

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


pub type MoveEvaluation<Spec> = <<Spec as MCTS>::TreePolicy as TreePolicy<Spec>>::MoveEvaluation;
pub type StateEvaluation<Spec> = <<Spec as MCTS>::Eval as Evaluator<Spec>>::StateEvaluation;
pub type Move<Spec> = <<Spec as MCTS>::State as GameState>::Move;
pub type MoveList<Spec> = <<Spec as MCTS>::State as GameState>::MoveList;
pub type Player<Spec> = <<Spec as MCTS>::State as GameState>::Player;
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
	pub fn from_u8(v: u8) -> Self {
		match v {
			1 => ProvenValue::Win,
			2 => ProvenValue::Loss,
			3 => ProvenValue::Draw,
			_ => ProvenValue::Unknown,
		}
	}
}

pub trait GameState: Clone {
	type Move: Sync + Send + Clone;
	type Player: Sync;
	type MoveList: std::iter::IntoIterator<Item = Self::Move>;

	fn current_player(&self) -> Self::Player;
	fn available_moves(&self) -> Self::MoveList;
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

pub trait Evaluator<Spec: MCTS>: Sync {
	type StateEvaluation: Sync + Send;

	fn evaluate_new_state(
		&self,
		state: &Spec::State,
		moves: &MoveList<Spec>,
		handle: Option<SearchHandle<Spec>>,
	) -> (Vec<MoveEvaluation<Spec>>, Self::StateEvaluation);

	fn evaluate_existing_state(
		&self,
		state: &Spec::State,
		existing_evaln: &Self::StateEvaluation,
		handle: SearchHandle<Spec>,
	) -> Self::StateEvaluation;

	fn interpret_evaluation_for_player(&self, evaluation: &Self::StateEvaluation, player: &Player<Spec>) -> i64;
}

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
	pub fn playout_n(&mut self, n: u64) {
		for _ in 0..n {
			self.playout();
		}
	}
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
	pub fn principal_variation_info(&self, num_moves: usize) -> Vec<MoveInfoHandle<'_, Spec>> {
		self.search_tree.principal_variation(num_moves)
	}
	pub fn principal_variation(&self, num_moves: usize) -> Vec<Move<Spec>> {
		self.search_tree
			.principal_variation(num_moves)
			.into_iter()
			.map(|x| x.get_move().clone())
			.collect()
	}
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
	pub fn tree(&self) -> &SearchTree<Spec> {
		&self.search_tree
	}
	/// Returns the proven value of the root node (for MCTS-Solver).
	pub fn root_proven_value(&self) -> ProvenValue {
		self.search_tree.root_proven_value()
	}
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

pub enum CycleBehaviour<Spec: MCTS> {
	Ignore,
	UseCurrentEvalWhenCycleDetected,
	PanicWhenCycleDetected,
	UseThisEvalWhenCycleDetected(StateEvaluation<Spec>),
}
