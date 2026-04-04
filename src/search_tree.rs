use rand::{rngs::SmallRng, Rng, SeedableRng};

use {
	super::*,
	atomics::*,
	std::{
		fmt,
		fmt::{Debug, Display, Formatter},
		ptr::null_mut,
		sync::Mutex,
	},
};

use tree_policy::TreePolicy;

/// You're not intended to use this class (use an `MCTSManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree<Spec: MCTS> {
	root_node: SearchNode<Spec>,
	root_state: Spec::State,
	tree_policy: Spec::TreePolicy,
	table: Spec::TranspositionTable,
	eval: Spec::Eval,
	manager: Spec,

	num_nodes: AtomicUsize,
	thread_counter: AtomicUsize,
	orphaned: Mutex<Vec<Box<SearchNode<Spec>>>>,
	transposition_table_hits: AtomicUsize,
	delayed_transposition_table_hits: AtomicUsize,
	expansion_contention_events: AtomicUsize,
}

struct NodeStats {
	visits: AtomicUsize,
	sum_evaluations: AtomicI64,
}

/// Information about a single move from a search node: the move, its evaluation,
/// visit statistics, and a pointer to the child node.
pub struct MoveInfo<Spec: MCTS> {
	mov: Move<Spec>,
	move_evaluation: MoveEvaluation<Spec>,
	child: AtomicPtr<SearchNode<Spec>>,
	owned: AtomicBool,
	stats: NodeStats,
}

/// A node in the search tree. Contains the list of moves, evaluation,
/// visit statistics, and solver/bounds state.
pub struct SearchNode<Spec: MCTS> {
	moves: Vec<MoveInfo<Spec>>,
	data: Spec::NodeData,
	evaln: StateEvaluation<Spec>,
	stats: NodeStats,
	proven: AtomicU8,
	score_lower: AtomicI32,
	score_upper: AtomicI32,
	is_chance: bool,
	chance_probs: Vec<f64>,
}

impl<Spec: MCTS> SearchNode<Spec> {
	fn new(moves: Vec<MoveInfo<Spec>>, evaln: StateEvaluation<Spec>) -> Self {
		Self {
			moves,
			data: Default::default(),
			evaln,
			stats: NodeStats::new(),
			proven: AtomicU8::new(ProvenValue::Unknown as u8),
			score_lower: AtomicI32::new(i32::MIN),
			score_upper: AtomicI32::new(i32::MAX),
			is_chance: false,
			chance_probs: Vec::new(),
		}
	}

	/// The proven game-theoretic value of this node (for MCTS-Solver).
	pub fn proven_value(&self) -> ProvenValue {
		ProvenValue::from_u8(self.proven.load(Ordering::Relaxed))
	}

	/// The proven score bounds of this node (for Score-Bounded MCTS).
	pub fn score_bounds(&self) -> ScoreBounds {
		ScoreBounds {
			lower: self.score_lower.load(Ordering::Relaxed),
			upper: self.score_upper.load(Ordering::Relaxed),
		}
	}
}

impl<Spec: MCTS> MoveInfo<Spec> {
	fn new(mov: Move<Spec>, move_evaluation: MoveEvaluation<Spec>) -> Self {
		MoveInfo {
			mov,
			move_evaluation,
			child: AtomicPtr::default(),
			stats: NodeStats::new(),
			owned: AtomicBool::new(false),
		}
	}

	/// The move this edge represents.
	pub fn get_move(&self) -> &Move<Spec> {
		&self.mov
	}

	/// The tree policy's evaluation of this move (e.g., prior probability).
	pub fn move_evaluation(&self) -> &MoveEvaluation<Spec> {
		&self.move_evaluation
	}

	pub(crate) fn set_move_evaluation(&mut self, eval: MoveEvaluation<Spec>) {
		self.move_evaluation = eval;
	}

	/// Number of times this move has been selected during search.
	pub fn visits(&self) -> u64 {
		self.stats.visits.load(Ordering::Relaxed) as u64
	}

	/// Sum of backpropagated rewards through this move.
	pub fn sum_rewards(&self) -> i64 {
		self.stats.sum_evaluations.load(Ordering::Relaxed)
	}

	/// The child node reached by this move, if expanded.
	pub fn child(&self) -> Option<NodeHandle<'_, Spec>> {
		let ptr = self.child.load(Ordering::Relaxed);
		if ptr.is_null() {
			None
		} else {
			unsafe { Some(NodeHandle { node: &*ptr }) }
		}
	}

	/// Returns the proven value of this move's child node.
	/// Returns `ProvenValue::Unknown` if the child has not been expanded.
	/// The value is from the child's mover perspective (inverted from parent's).
	pub fn child_proven_value(&self) -> ProvenValue {
		let ptr = self.child.load(Ordering::Relaxed);
		if ptr.is_null() {
			ProvenValue::Unknown
		} else {
			unsafe { (*ptr).proven_value() }
		}
	}

	/// Returns the score bounds of this move's child node.
	/// Returns `ScoreBounds::UNBOUNDED` if the child has not been expanded.
	/// Bounds are from the child's current player's perspective.
	pub fn child_score_bounds(&self) -> ScoreBounds {
		let ptr = self.child.load(Ordering::Relaxed);
		if ptr.is_null() {
			ScoreBounds::UNBOUNDED
		} else {
			unsafe { (*ptr).score_bounds() }
		}
	}
}

impl<Spec: MCTS> Display for MoveInfo<Spec>
where
	Move<Spec>: Display,
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		let own_str = if self.owned.load(Ordering::Relaxed) {
			""
		} else {
			" [child pointer is alias]"
		};
		if self.visits() == 0 {
			write!(f, "{} [0 visits]{}", self.mov, own_str)
		} else {
			write!(
				f,
				"{} [{} visit{}] [{} avg reward]{}",
				self.mov,
				self.visits(),
				if self.visits() == 1 { "" } else { "s" },
				self.sum_rewards() as f64 / self.visits() as f64,
				own_str
			)
		}
	}
}

impl<Spec: MCTS> Debug for MoveInfo<Spec>
where
	Move<Spec>: Debug,
{
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		let own_str = if self.owned.load(Ordering::Relaxed) {
			""
		} else {
			" [child pointer is alias]"
		};
		if self.visits() == 0 {
			write!(f, "{:?} [0 visits]{}", self.mov, own_str)
		} else {
			write!(
				f,
				"{:?} [{} visit{}] [{} avg reward]{}",
				self.mov,
				self.visits(),
				if self.visits() == 1 { "" } else { "s" },
				self.sum_rewards() as f64 / self.visits() as f64,
				own_str
			)
		}
	}
}

impl<Spec: MCTS> Drop for MoveInfo<Spec> {
	fn drop(&mut self) {
		if !self.owned.load(Ordering::SeqCst) {
			return;
		}
		let ptr = self.child.load(Ordering::SeqCst);
		if !ptr.is_null() {
			unsafe {
				drop(Box::from_raw(ptr));
			}
		}
	}
}

fn create_node<Spec: MCTS>(
	eval: &Spec::Eval,
	policy: &Spec::TreePolicy,
	state: &Spec::State,
	handle: Option<SearchHandle<Spec>>,
	solver_enabled: bool,
	score_bounded: bool,
	closed_loop: bool,
) -> SearchNode<Spec> {
	// Closed-loop chance: if the state has pending chance outcomes, create a chance node
	if closed_loop {
		if let Some(outcomes) = state.chance_outcomes() {
			let probs: Vec<f64> = outcomes.iter().map(|(_, p)| *p).collect();
			// Get state evaluation from evaluator (using available_moves, likely empty)
			let avail = state.available_moves();
			let (_, state_eval) = eval.evaluate_new_state(state, &avail, handle);
			let moves: Vec<MoveInfo<Spec>> = outcomes
				.into_iter()
				.map(|(m, _)| MoveInfo::new(m, Default::default()))
				.collect();
			return SearchNode {
				moves,
				data: Default::default(),
				evaln: state_eval,
				stats: NodeStats::new(),
				proven: AtomicU8::new(ProvenValue::Unknown as u8),
				score_lower: AtomicI32::new(i32::MIN),
				score_upper: AtomicI32::new(i32::MAX),
				is_chance: true,
				chance_probs: probs,
			};
		}
	}

	let moves = state.available_moves();
	let (move_eval, state_eval) = eval.evaluate_new_state(state, &moves, handle);
	policy.validate_evaluations(&move_eval);
	let mut moves: Vec<MoveInfo<Spec>> = moves
		.into_iter()
		.zip(move_eval)
		.map(|(m, e)| MoveInfo::new(m, e))
		.collect();
	moves.sort_by(|a, b| policy.compare_move_evaluations(&a.move_evaluation, &b.move_evaluation));
	let node = SearchNode::new(moves, state_eval);
	if node.moves.is_empty() {
		let tv = state.terminal_value();
		let ts = state.terminal_score();

		// Debug-mode consistency check when both are provided
		#[cfg(debug_assertions)]
		if let (Some(pv), Some(score)) = (tv, ts) {
			match pv {
				ProvenValue::Win => debug_assert!(score > 0,
					"terminal_value is Win but terminal_score is {score}"),
				ProvenValue::Loss => debug_assert!(score < 0,
					"terminal_value is Loss but terminal_score is {score}"),
				ProvenValue::Draw => debug_assert!(score == 0,
					"terminal_value is Draw but terminal_score is {score}"),
				ProvenValue::Unknown => {}
			}
		}

		if solver_enabled {
			let proven = tv.or_else(|| {
				// Derive from terminal_score when terminal_value is not provided
				ts.map(|s| if s > 0 { ProvenValue::Win } else if s < 0 { ProvenValue::Loss } else { ProvenValue::Draw })
			});
			if let Some(pv) = proven {
				node.proven.store(pv as u8, Ordering::Relaxed);
			}
		}
		if score_bounded {
			let score = ts.or_else(|| {
				// Derive from terminal_value when terminal_score is not provided
				tv.and_then(|pv| match pv {
					ProvenValue::Win => Some(1),
					ProvenValue::Loss => Some(-1),
					ProvenValue::Draw => Some(0),
					ProvenValue::Unknown => None, // Don't derive bounds from Unknown
				})
			});
			if let Some(s) = score {
				node.score_lower.store(s, Ordering::Relaxed);
				node.score_upper.store(s, Ordering::Relaxed);
			}
		}
	}
	node
}

/// Attempt to determine a proven value for a node by examining all its children.
/// Returns Unknown if the node cannot be proven yet.
///
/// Convention: proven values are from the current_player's perspective at each node.
/// - Child's Loss (opponent loses) → parent can Win by choosing this child
/// - Child's Win (opponent wins) → this move is bad for the parent
/// - Parent is Loss only if ALL children are Win (no escape)
/// - Parent is Win if ANY child is Loss (parent picks the winning move)
fn try_prove_node<Spec: MCTS>(node: &SearchNode<Spec>) -> ProvenValue {
	if node.moves.is_empty() {
		return node.proven_value();
	}

	let mut all_children_proven = true;
	let mut all_children_win = true; // all Win from child's perspective = all bad for parent
	let mut has_child_draw = false;

	for move_info in &node.moves {
		let child_ptr = move_info.child.load(Ordering::Relaxed);
		if child_ptr.is_null() {
			return ProvenValue::Unknown;
		}
		let child_proven = unsafe { (*child_ptr).proven_value() };
		match child_proven {
			ProvenValue::Unknown => {
				all_children_proven = false;
				all_children_win = false;
			}
			ProvenValue::Win => {
				// Child's current_player wins → bad for parent (this move loses)
			}
			ProvenValue::Loss => {
				// Child's current_player loses → parent can win by choosing this!
				return ProvenValue::Win;
			}
			ProvenValue::Draw => {
				has_child_draw = true;
				all_children_win = false;
			}
		}
	}

	if all_children_proven && all_children_win {
		// Every move leads to the opponent winning → parent loses
		return ProvenValue::Loss;
	}

	if all_children_proven && has_child_draw {
		// All children are proven, best outcome is a draw
		return ProvenValue::Draw;
	}

	ProvenValue::Unknown
}

/// Compute tightened score bounds for a node by examining all children (negamax).
/// Parent's lower = max over children of negate(child.upper).
/// Parent's upper = max over children of negate(child.lower).
fn try_tighten_bounds<Spec: MCTS>(node: &SearchNode<Spec>) -> ScoreBounds {
	if node.moves.is_empty() {
		return node.score_bounds();
	}

	let mut best_lower = i32::MIN;
	let mut best_upper = i32::MIN;

	for move_info in &node.moves {
		let child_ptr = move_info.child.load(Ordering::Relaxed);
		let (child_lower, child_upper) = if child_ptr.is_null() {
			(i32::MIN, i32::MAX)
		} else {
			unsafe {
				let child = &*child_ptr;
				(
					child.score_lower.load(Ordering::Relaxed),
					child.score_upper.load(Ordering::Relaxed),
				)
			}
		};

		// From parent's perspective (negamax): [-child_upper, -child_lower]
		let parent_lower_from_child = negate_bound(child_upper);
		let parent_upper_from_child = negate_bound(child_lower);

		best_lower = best_lower.max(parent_lower_from_child);
		best_upper = best_upper.max(parent_upper_from_child);
	}

	ScoreBounds {
		lower: best_lower,
		upper: best_upper,
	}
}

/// Prove a chance node: all children must be proven.
/// WIN only if all outcomes WIN, LOSS only if all LOSS. No negation (same player).
fn try_prove_chance_node<Spec: MCTS>(node: &SearchNode<Spec>) -> ProvenValue {
	if node.moves.is_empty() {
		return node.proven_value();
	}
	let mut all_win = true;
	let mut all_loss = true;
	for mi in &node.moves {
		let ptr = mi.child.load(Ordering::Relaxed);
		if ptr.is_null() {
			return ProvenValue::Unknown;
		}
		let child_proven = unsafe { (*ptr).proven_value() };
		match child_proven {
			ProvenValue::Unknown => return ProvenValue::Unknown,
			ProvenValue::Win => { all_loss = false; }
			ProvenValue::Loss => { all_win = false; }
			ProvenValue::Draw => { all_win = false; all_loss = false; }
		}
	}
	if all_win {
		ProvenValue::Win
	} else if all_loss {
		ProvenValue::Loss
	} else {
		ProvenValue::Draw
	}
}

/// Tighten score bounds at a chance node using weighted averages.
/// No negation (same player perspective through chance events).
fn try_tighten_bounds_chance<Spec: MCTS>(node: &SearchNode<Spec>) -> ScoreBounds {
	if node.moves.is_empty() {
		return node.score_bounds();
	}
	let mut lower_sum: f64 = 0.0;
	let mut upper_sum: f64 = 0.0;

	for (mi, &prob) in node.moves.iter().zip(node.chance_probs.iter()) {
		let ptr = mi.child.load(Ordering::Relaxed);
		if ptr.is_null() {
			return ScoreBounds::UNBOUNDED;
		}
		let child_lower = unsafe { (*ptr).score_lower.load(Ordering::Relaxed) };
		let child_upper = unsafe { (*ptr).score_upper.load(Ordering::Relaxed) };
		if child_lower == i32::MIN || child_upper == i32::MAX {
			return ScoreBounds::UNBOUNDED;
		}
		lower_sum += prob * child_lower as f64;
		upper_sum += prob * child_upper as f64;
	}

	ScoreBounds {
		lower: lower_sum.floor() as i32,
		upper: upper_sum.ceil() as i32,
	}
}

/// Sample a child from a chance node by probability.
fn sample_chance_child<'a, Spec: MCTS>(
	node: &'a SearchNode<Spec>,
	rng: &mut SmallRng,
) -> &'a MoveInfo<Spec> {
	debug_assert!(node.is_chance);
	debug_assert!(!node.moves.is_empty(), "chance node has no outcomes");
	let roll: f64 = rng.gen();
	let mut cumulative = 0.0;
	for (mi, &prob) in node.moves.iter().zip(node.chance_probs.iter()) {
		cumulative += prob;
		if roll < cumulative {
			return mi;
		}
	}
	node.moves.last().unwrap()
}

fn is_cycle<T>(past: &[&T], current: &T) -> bool {
	past.iter().any(|x| std::ptr::eq(*x, current))
}

/// Sample from a weighted distribution of chance outcomes.
fn sample_chance_outcome<'a, M>(outcomes: &'a [(M, f64)], rng: &mut SmallRng) -> &'a M {
	debug_assert!(!outcomes.is_empty(), "chance_outcomes returned empty vec");
	let roll: f64 = rng.gen();
	let mut cumulative = 0.0;
	for (outcome, prob) in outcomes {
		cumulative += prob;
		if roll < cumulative {
			return outcome;
		}
	}
	// Floating-point rounding: return last outcome
	&outcomes.last().unwrap().0
}

impl<Spec: MCTS> SearchTree<Spec> {
	/// Create a new search tree rooted at the given state.
	pub fn new(
		state: Spec::State,
		manager: Spec,
		tree_policy: Spec::TreePolicy,
		eval: Spec::Eval,
		table: Spec::TranspositionTable,
	) -> Self {
		let solver = manager.solver_enabled();
		let score_bounded = manager.score_bounded_enabled();
		let closed_loop = manager.closed_loop_chance();
		let mut root_node = create_node(&eval, &tree_policy, &state, None, solver, score_bounded, closed_loop);
		if let Some((epsilon, alpha)) = manager.dirichlet_noise() {
			let mut rng = match manager.rng_seed() {
				Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(u64::MAX)),
				None => SmallRng::from_rng(rand::thread_rng()).unwrap(),
			};
			tree_policy.apply_dirichlet_noise(&mut root_node.moves, epsilon, alpha, &mut rng);
		}
		Self {
			root_state: state,
			root_node,
			manager,
			tree_policy,
			eval,
			table,
			num_nodes: 1.into(),
			thread_counter: 0.into(),
			orphaned: Mutex::new(Vec::new()),
			transposition_table_hits: 0.into(),
			delayed_transposition_table_hits: 0.into(),
			expansion_contention_events: 0.into(),
		}
	}

	/// Create thread-local data, seeded if the MCTS config provides a seed.
	pub fn make_thread_data(&self) -> ThreadDataFull<Spec>
	where
		ThreadData<Spec>: Default,
	{
		let mut tld = ThreadDataFull::<Spec>::default();
		if let Some(base_seed) = self.manager.rng_seed() {
			let thread_id = self.thread_counter.fetch_add(1, Ordering::Relaxed) as u64;
			let seed = base_seed.wrapping_add(thread_id);
			self.tree_policy.seed_thread_data(&mut tld.tld.policy_data, seed);
			// Seed chance RNG with a different offset to avoid correlation
			tld.chance_rng = SmallRng::seed_from_u64(seed.wrapping_add(0xCAFE_BABE));
		}
		tld
	}

	/// Reset the tree, re-creating the root node from the original state.
	pub fn reset(self) -> Self {
		Self::new(self.root_state, self.manager, self.tree_policy, self.eval, self.table)
	}

	/// The MCTS configuration.
	pub fn spec(&self) -> &Spec {
		&self.manager
	}

	/// Number of nodes currently in the tree.
	pub fn num_nodes(&self) -> usize {
		self.num_nodes.load(Ordering::SeqCst)
	}

	/// Run a single playout from root to leaf. Returns `false` if the node limit is reached or root is proven.
	#[inline(never)]
	pub fn playout(&self, tld: &mut ThreadDataFull<Spec>) -> bool {
		let sentinel = IncreaseSentinel::new(&self.num_nodes);
		if sentinel.num_nodes >= self.manager.node_limit() {
			return false;
		}
		let solver = self.manager.solver_enabled();
		let score_bounded = self.manager.score_bounded_enabled();
		// If root is already proven, stop searching
		if solver && self.root_node.proven_value() != ProvenValue::Unknown {
			return false;
		}
		// If root score bounds have converged, stop searching
		if score_bounded {
			let bounds = self.root_node.score_bounds();
			if bounds.is_proven() {
				return false;
			}
		}
		let mut state = self.root_state.clone();
		let path: &mut Vec<&MoveInfo<Spec>> = &mut tld.path.reuse_allocation();
		let node_path: &mut Vec<&SearchNode<Spec>> = &mut tld.node_path.reuse_allocation();
		let players: &mut Vec<Player<Spec>> = &mut tld.players.reuse_allocation();
		let chance_rng = &mut tld.chance_rng;
		let closed_loop = self.manager.closed_loop_chance();
		let tld = &mut tld.tld;

		// Resolve any pending chance events at the root state (open-loop only;
		// closed-loop handles them as tree nodes)
		if !closed_loop {
			while let Some(outcomes) = state.chance_outcomes() {
				let outcome = sample_chance_outcome(&outcomes, chance_rng);
				state.make_move(outcome);
			}
		}

		let mut did_we_create = false;
		let mut node = &self.root_node;
		loop {
			if node.moves.is_empty() {
				break;
			}
			if solver && node.proven_value() != ProvenValue::Unknown {
				break;
			}
			if path.len() >= self.manager.max_playout_depth() {
				break;
			}
			if path.len() >= self.manager.max_playout_length() {
				break;
			}
			// Select child: probability sampling for chance nodes, tree policy for decision nodes
			let choice = if node.is_chance {
				sample_chance_child(node, chance_rng)
			} else {
				let parent_visits = node.stats.visits.load(Ordering::Relaxed) as u64;
				let max_children = state.max_children(parent_visits).max(1).min(node.moves.len());
				self.tree_policy
					.choose_child(node.moves[..max_children].iter(), self.make_handle(node, tld))
			};
			choice.stats.down(&self.manager);
			players.push(state.current_player());
			path.push(choice);
			assert!(
				path.len() <= self.manager.max_playout_length(),
				"playout length exceeded maximum of {} (maybe the transposition table is creating an infinite loop?)",
				self.manager.max_playout_length()
			);
			state.make_move(&choice.mov);
			// Resolve any chance events after the move (open-loop only)
			if !closed_loop {
				while let Some(outcomes) = state.chance_outcomes() {
					let outcome = sample_chance_outcome(&outcomes, chance_rng);
					state.make_move(outcome);
				}
			}
			let (new_node, new_did_we_create) = self.descend(&state, choice, node, tld);
			node = new_node;
			did_we_create = new_did_we_create;
			match self.manager.cycle_behaviour() {
				CycleBehaviour::Ignore => (),
				CycleBehaviour::PanicWhenCycleDetected => {
					if is_cycle(node_path, node) {
						panic!("cycle detected! you should do one of the following:\n- make states acyclic\n- remove transposition table\n- change cycle_behaviour()");
					}
				}
				CycleBehaviour::UseCurrentEvalWhenCycleDetected => {
					if is_cycle(node_path, node) {
						break;
					}
				}
				CycleBehaviour::UseThisEvalWhenCycleDetected(e) => {
					if is_cycle(node_path, node) {
						self.finish_playout(path, node_path, players, tld, &e);
						return true;
					}
				}
			};
			node_path.push(node);
			node.stats.down(&self.manager);
			if node.stats.visits.load(Ordering::Relaxed) as u64 <= self.manager.visits_before_expansion() {
				break;
			}
		}
		let new_evaln = if did_we_create {
			None
		} else {
			Some(
				self.eval
					.evaluate_existing_state(&state, &node.evaln, self.make_handle(node, tld)),
			)
		};
		let evaln = new_evaln.as_ref().unwrap_or(&node.evaln);
		self.finish_playout(path, node_path, players, tld, evaln);
		true
	}

	fn descend<'a, 'b>(
		&'a self,
		state: &Spec::State,
		choice: &MoveInfo<Spec>,
		current_node: &'b SearchNode<Spec>,
		tld: &'b mut ThreadData<Spec>,
	) -> (&'a SearchNode<Spec>, bool) {
		let child = choice.child.load(Ordering::Relaxed);
		if !child.is_null() {
			return unsafe { (&*child, false) };
		}
		if let Some(node) = self.table.lookup(state, self.make_handle(current_node, tld)) {
			let child = choice
				.child
				.compare_exchange(null_mut(), node as *const _ as *mut _, Ordering::Relaxed, Ordering::Relaxed)
				.unwrap_or_else(|x| x);
			if child.is_null() {
				self.transposition_table_hits.fetch_add(1, Ordering::Relaxed);
				return (node, false);
			} else {
				return unsafe { (&*child, false) };
			}
		}
		let created = create_node(
			&self.eval,
			&self.tree_policy,
			state,
			Some(self.make_handle(current_node, tld)),
			self.manager.solver_enabled(),
			self.manager.score_bounded_enabled(),
			self.manager.closed_loop_chance(),
		);
		let created = Box::into_raw(Box::new(created));
		let other_child = choice.child.compare_exchange(null_mut(), created, Ordering::Relaxed, Ordering::Relaxed)
			.unwrap_or_else(|x| x);
		if !other_child.is_null() {
			self.expansion_contention_events.fetch_add(1, Ordering::Relaxed);
			unsafe {
				drop(Box::from_raw(created));
				return (&*other_child, false);
			}
		}
		if let Some(existing) = self
			.table
			.insert(state, unsafe { &*created }, self.make_handle(current_node, tld))
		{
			self.delayed_transposition_table_hits.fetch_add(1, Ordering::Relaxed);
			let existing_ptr = existing as *const _ as *mut _;
			choice.child.store(existing_ptr, Ordering::Relaxed);
			self.orphaned.lock().unwrap().push(unsafe { Box::from_raw(created) });
			return (existing, false);
		}
		choice.owned.store(true, Ordering::Relaxed);
		self.num_nodes.fetch_add(1, Ordering::Relaxed);
		unsafe { (&*created, true) }
	}

	fn finish_playout(
		&self,
		path: &[&MoveInfo<Spec>],
		node_path: &[&SearchNode<Spec>],
		players: &[Player<Spec>],
		tld: &mut ThreadData<Spec>,
		evaln: &StateEvaluation<Spec>,
	) {
		for ((move_info, player), node) in path.iter().zip(players.iter()).zip(node_path.iter()).rev() {
			let evaln_value = self.eval.interpret_evaluation_for_player(evaln, player);
			node.stats.up(&self.manager, evaln_value);
			move_info.stats.replace(&node.stats);
			// SAFETY: move_info.child is guaranteed non-null here because it was
			// populated by descend() during the playout traversal that built this path.
			unsafe {
				self.manager
					.on_backpropagation(evaln, self.make_handle(&*move_info.child.load(Ordering::Relaxed), tld));
			}
		}
		self.manager
			.on_backpropagation(evaln, self.make_handle(&self.root_node, tld));

		// Solver: propagate proven values bottom-up
		if self.manager.solver_enabled() {
			self.propagate_proven(path, node_path);
		}
		// Score-Bounded: propagate score bounds bottom-up
		if self.manager.score_bounded_enabled() {
			self.propagate_score_bounds(path, node_path);
		}
	}

	/// Walk the playout path bottom-up, attempting to prove nodes.
	fn propagate_proven(
		&self,
		path: &[&MoveInfo<Spec>],
		node_path: &[&SearchNode<Spec>],
	) {
		// Check each node along the path, starting from the deepest
		for i in (0..path.len()).rev() {
			// The child reached by path[i] is at node_path[i] (if it exists)
			let child_ptr = path[i].child.load(Ordering::Relaxed);
			if child_ptr.is_null() {
				break;
			}
			let child_proven = unsafe { (*child_ptr).proven_value() };
			if child_proven == ProvenValue::Unknown {
				break;
			}

			// Try to prove the parent node
			let parent = if i == 0 {
				&self.root_node
			} else {
				node_path[i - 1]
			};

			// Skip if parent is already proven
			if parent.proven_value() != ProvenValue::Unknown {
				continue;
			}

			let parent_proven = if parent.is_chance {
				try_prove_chance_node(parent)
			} else {
				try_prove_node(parent)
			};
			if parent_proven == ProvenValue::Unknown {
				break;
			}

			// Atomically set (Unknown → proven), ignore if already set
			let _ = parent.proven.compare_exchange(
				ProvenValue::Unknown as u8,
				parent_proven as u8,
				Ordering::Relaxed,
				Ordering::Relaxed,
			);
		}
	}

	/// Walk the playout path bottom-up, tightening score bounds.
	fn propagate_score_bounds(
		&self,
		path: &[&MoveInfo<Spec>],
		node_path: &[&SearchNode<Spec>],
	) {
		for i in (0..path.len()).rev() {
			let parent = if i == 0 {
				&self.root_node
			} else {
				node_path[i - 1]
			};

			let new_bounds = if parent.is_chance {
				try_tighten_bounds_chance(parent)
			} else {
				try_tighten_bounds(parent)
			};

			let old_lower = parent.score_lower.load(Ordering::Relaxed);
			let old_upper = parent.score_upper.load(Ordering::Relaxed);

			// Monotonically tighten lower (only increases)
			if new_bounds.lower > old_lower {
				let _ = parent.score_lower.compare_exchange_weak(
					old_lower,
					new_bounds.lower,
					Ordering::Relaxed,
					Ordering::Relaxed,
				);
			}

			// Monotonically tighten upper (only decreases)
			if new_bounds.upper < old_upper {
				let _ = parent.score_upper.compare_exchange_weak(
					old_upper,
					new_bounds.upper,
					Ordering::Relaxed,
					Ordering::Relaxed,
				);
			}

			// Cross-system: converged bounds set proven value when solver is active
			if self.manager.solver_enabled() && new_bounds.lower == new_bounds.upper {
				let pv = if new_bounds.lower > 0 {
					ProvenValue::Win
				} else if new_bounds.lower < 0 {
					ProvenValue::Loss
				} else {
					ProvenValue::Draw
				};
				let _ = parent.proven.compare_exchange(
					ProvenValue::Unknown as u8,
					pv as u8,
					Ordering::Relaxed,
					Ordering::Relaxed,
				);
			}

			// If bounds didn't change, stop propagating
			if new_bounds.lower <= old_lower && new_bounds.upper >= old_upper {
				break;
			}
		}
	}

	fn make_handle<'a>(&'a self, node: &'a SearchNode<Spec>, tld: &'a mut ThreadData<Spec>) -> SearchHandle<'a, Spec> {
		SearchHandle {
			node,
			tld,
			manager: &self.manager,
		}
	}

	/// The game state at the root of the tree.
	pub fn root_state(&self) -> &Spec::State {
		&self.root_state
	}
	/// A handle to the root node.
	pub fn root_node(&self) -> NodeHandle<'_, Spec> {
		NodeHandle { node: &self.root_node }
	}

	/// The proven value of the root (for MCTS-Solver).
	pub fn root_proven_value(&self) -> ProvenValue {
		self.root_node.proven_value()
	}

	/// The score bounds of the root (for Score-Bounded MCTS).
	pub fn root_score_bounds(&self) -> ScoreBounds {
		self.root_node.score_bounds()
	}

	/// The best sequence of moves found by search, as move info handles.
	pub fn principal_variation(&self, num_moves: usize) -> Vec<MoveInfoHandle<'_, Spec>> {
		let mut result = Vec::new();
		let mut crnt = &self.root_node;
		while !crnt.moves.is_empty() && result.len() < num_moves {
			// Chance nodes: select by most-visited (reflects highest probability).
			// Decision nodes: use solver/bounds-aware selection.
			let choice = if crnt.is_chance {
				crnt.moves.iter().max_by_key(|c| c.visits()).unwrap()
			} else {
				self.manager.select_child_after_search(&crnt.moves)
			};
			result.push(choice);
			let child = choice.child.load(Ordering::SeqCst) as *const SearchNode<Spec>;
			if child.is_null() {
				break;
			} else {
				unsafe {
					crnt = &*child;
				}
			}
		}
		result
	}

	/// Diagnostic string with node counts, transposition hits, and contention events.
	pub fn diagnose(&self) -> String {
		let mut s = String::new();
		s.push_str(&format!(
			"{} nodes\n",
			thousands_separate(self.num_nodes.load(Ordering::Relaxed))
		));
		s.push_str(&format!(
			"{} transposition table hits\n",
			thousands_separate(self.transposition_table_hits.load(Ordering::Relaxed))
		));
		s.push_str(&format!(
			"{} delayed transposition table hits\n",
			thousands_separate(self.delayed_transposition_table_hits.load(Ordering::Relaxed))
		));
		s.push_str(&format!(
			"{} expansion contention events\n",
			thousands_separate(self.expansion_contention_events.load(Ordering::Relaxed))
		));
		s.push_str(&format!("{} orphaned nodes\n", self.orphaned.lock().unwrap().len()));
		s
	}
}

/// A borrowed reference to a `MoveInfo` in the search tree.
pub type MoveInfoHandle<'a, Spec> = &'a MoveInfo<Spec>;

/// Summary statistics for a root child, returned by `root_child_stats()`.
pub struct ChildStats<Spec: MCTS> {
	pub mov: Move<Spec>,
	pub visits: u64,
	pub avg_reward: f64,
	pub move_evaluation: MoveEvaluation<Spec>,
	pub proven_value: ProvenValue,
	pub score_bounds: ScoreBounds,
}

impl<Spec: MCTS> SearchTree<Spec>
where
	MoveEvaluation<Spec>: Clone,
{
	/// Visit counts, average rewards, and proven values for all root children.
	pub fn root_child_stats(&self) -> Vec<ChildStats<Spec>> {
		self.root_node
			.moves
			.iter()
			.map(|mi| {
				let visits = mi.visits();
				let avg_reward = if visits == 0 {
					0.0
				} else {
					mi.sum_rewards() as f64 / visits as f64
				};
				ChildStats {
					mov: mi.get_move().clone(),
					visits,
					avg_reward,
					move_evaluation: mi.move_evaluation().clone(),
					proven_value: mi.child_proven_value(),
					score_bounds: mi.child_score_bounds(),
				}
			})
			.collect()
	}
}

impl<Spec: MCTS> SearchTree<Spec>
where
	Move<Spec>: Debug,
{
	/// Print root moves sorted by visit count (Debug format).
	pub fn debug_moves(&self) {
		let mut moves: Vec<&MoveInfo<Spec>> = self.root_node.moves.iter().collect();
		moves.sort_by_key(|x| -(x.visits() as i64));
		for mov in moves {
			println!("{:?}", mov);
		}
	}
}

impl<Spec: MCTS> SearchTree<Spec>
where
	Move<Spec>: Display,
{
	/// Print root moves sorted by visit count (Display format).
	pub fn display_moves(&self) {
		let mut moves: Vec<&MoveInfo<Spec>> = self.root_node.moves.iter().collect();
		moves.sort_by_key(|x| -(x.visits() as i64));
		for mov in moves {
			println!("{}", mov);
		}
	}
}

/// Error returned when `advance_root()` fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdvanceError {
	/// The move does not exist among root children.
	MoveNotFound,
	/// The child node was never expanded during search.
	ChildNotExpanded,
	/// The child is a transposition table alias and cannot be detached.
	ChildNotOwned,
}

impl Display for AdvanceError {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		match self {
			AdvanceError::MoveNotFound => write!(f, "move not found in root children"),
			AdvanceError::ChildNotExpanded => write!(f, "child node was never expanded"),
			AdvanceError::ChildNotOwned => write!(f, "child node is a transposition alias (not owned)"),
		}
	}
}

impl<Spec: MCTS> SearchTree<Spec>
where
	Move<Spec>: PartialEq,
{
	/// Advance the root to a child node, preserving the subtree below. Clears the transposition table.
	pub fn advance_root(&mut self, mov: &Move<Spec>) -> Result<(), AdvanceError> {
		// Find the MoveInfo matching the chosen move
		let idx = self
			.root_node
			.moves
			.iter()
			.position(|m| m.mov == *mov)
			.ok_or(AdvanceError::MoveNotFound)?;

		// Validate the child exists and is owned
		let move_info = &self.root_node.moves[idx];
		let child_ptr = move_info.child.load(Ordering::SeqCst);
		if child_ptr.is_null() {
			return Err(AdvanceError::ChildNotExpanded);
		}
		if !move_info.owned.load(Ordering::SeqCst) {
			return Err(AdvanceError::ChildNotOwned);
		}

		// Detach the child from its MoveInfo to prevent double-free
		self.root_node.moves[idx]
			.child
			.store(null_mut(), Ordering::SeqCst);
		self.root_node.moves[idx]
			.owned
			.store(false, Ordering::SeqCst);

		// Reconstruct the Box and extract the SearchNode
		let new_root = unsafe { *Box::from_raw(child_ptr) };

		// Advance the game state
		self.root_state.make_move(mov);

		// Swap in the new root, get the old one
		let old_root = std::mem::replace(&mut self.root_node, new_root);

		// Apply Dirichlet noise to the new root's priors
		if let Some((epsilon, alpha)) = self.manager.dirichlet_noise() {
			let mut rng = match self.manager.rng_seed() {
				Some(seed) => SmallRng::seed_from_u64(seed.wrapping_add(u64::MAX)),
				None => SmallRng::from_rng(rand::thread_rng()).unwrap(),
			};
			self.tree_policy
				.apply_dirichlet_noise(&mut self.root_node.moves, epsilon, alpha, &mut rng);
		}

		// Clear transposition table before dropping old root (prevent dangling pointers)
		self.table.clear();

		// Drop old root — MoveInfo Drop impls cascade-free sibling subtrees
		drop(old_root);

		// Clear any previously deferred orphaned nodes
		self.orphaned.lock().unwrap().clear();

		// Reset counters
		self.num_nodes.store(1, Ordering::SeqCst);
		self.transposition_table_hits.store(0, Ordering::SeqCst);
		self.delayed_transposition_table_hits.store(0, Ordering::SeqCst);
		self.expansion_contention_events.store(0, Ordering::SeqCst);

		Ok(())
	}
}

/// An immutable handle to a search node. Provides access to node data, moves, and solver state.
#[derive(Clone, Copy)]
pub struct NodeHandle<'a, Spec: 'a + MCTS> {
	node: &'a SearchNode<Spec>,
}

impl<'a, Spec: MCTS> NodeHandle<'a, Spec> {
	/// User-defined node data.
	pub fn data(&self) -> &'a Spec::NodeData {
		&self.node.data
	}
	/// Iterator over this node's moves.
	pub fn moves(&self) -> Moves<'_, Spec> {
		Moves {
			iter: self.node.moves.iter(),
		}
	}
	/// The proven game-theoretic value of this node.
	pub fn proven_value(&self) -> ProvenValue {
		self.node.proven_value()
	}
	/// The proven score bounds of this node.
	pub fn score_bounds(&self) -> ScoreBounds {
		self.node.score_bounds()
	}
	/// Convert to a raw pointer for external storage.
	pub fn into_raw(&self) -> *const () {
		self.node as *const _ as *const ()
	}
	/// # Safety
	/// `ptr` must have been obtained from `into_raw()` on a still-live `NodeHandle`.
	pub unsafe fn from_raw(ptr: *const ()) -> Self {
		NodeHandle {
			node: &*(ptr as *const SearchNode<Spec>),
		}
	}
}

/// Iterator over the moves of a search node.
#[derive(Clone)]
pub struct Moves<'a, Spec: 'a + MCTS> {
	iter: std::slice::Iter<'a, MoveInfo<Spec>>,
}

impl<'a, Spec: 'a + MCTS> Iterator for Moves<'a, Spec> {
	type Item = &'a MoveInfo<Spec>;
	fn next(&mut self) -> Option<Self::Item> {
		self.iter.next()
	}
}

/// A handle passed to evaluators and callbacks during search. Provides access to the
/// current node and thread-local data.
pub struct SearchHandle<'a, Spec: 'a + MCTS> {
	node: &'a SearchNode<Spec>,
	tld: &'a mut ThreadData<Spec>,
	manager: &'a Spec,
}

impl<'a, Spec: MCTS> SearchHandle<'a, Spec> {
	/// The current search node.
	pub fn node(&self) -> NodeHandle<'a, Spec> {
		NodeHandle { node: self.node }
	}
	/// Mutable access to thread-local data.
	pub fn thread_data(&mut self) -> &mut ThreadData<Spec> {
		self.tld
	}
	/// The MCTS configuration.
	pub fn mcts(&self) -> &'a Spec {
		self.manager
	}
}

impl NodeStats {
	fn new() -> Self {
		NodeStats {
			sum_evaluations: AtomicI64::new(0),
			visits: AtomicUsize::new(0),
		}
	}
	fn down<Spec: MCTS>(&self, manager: &Spec) {
		self.sum_evaluations
			.fetch_sub(manager.virtual_loss(), Ordering::Relaxed);
		self.visits.fetch_add(1, Ordering::Relaxed);
	}
	fn up<Spec: MCTS>(&self, manager: &Spec, evaln: i64) {
		let delta = evaln + manager.virtual_loss();
		self.sum_evaluations.fetch_add(delta, Ordering::Relaxed);
	}
	fn replace(&self, other: &NodeStats) {
		self.visits
			.store(other.visits.load(Ordering::Relaxed), Ordering::Relaxed);
		self.sum_evaluations
			.store(other.sum_evaluations.load(Ordering::Relaxed), Ordering::Relaxed);
	}
}

struct IncreaseSentinel<'a> {
	x: &'a AtomicUsize,
	num_nodes: usize,
}

impl<'a> IncreaseSentinel<'a> {
	fn new(x: &'a AtomicUsize) -> Self {
		let num_nodes = x.fetch_add(1, Ordering::Relaxed);
		Self { x, num_nodes }
	}
}

impl<'a> Drop for IncreaseSentinel<'a> {
	fn drop(&mut self) {
		self.x.fetch_sub(1, Ordering::Relaxed);
	}
}
