use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;
use std::time::Duration;

fn make_no_transposition_mcts() -> MCTSManager<NoTranspositionMCTS> {
	MCTSManager::new(
		CountingGame(0),
		NoTranspositionMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	)
}

// ---------------------------------------------------------------------------
// Game definitions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct CountingGame(i64);

#[derive(Clone, Debug, PartialEq)]
enum Move {
	Add,
	Sub,
}

impl GameState for CountingGame {
	type Move = Move;
	type Player = ();
	type MoveList = Vec<Move>;

	fn current_player(&self) -> Self::Player {}
	fn available_moves(&self) -> Vec<Move> {
		if self.0 == 100 {
			vec![]
		} else {
			vec![Move::Add, Move::Sub]
		}
	}
	fn make_move(&mut self, mov: &Self::Move) {
		match *mov {
			Move::Add => self.0 += 1,
			Move::Sub => self.0 -= 1,
		}
	}
}

impl TranspositionHash for CountingGame {
	fn hash(&self) -> u64 {
		self.0 as u64
	}
}

struct MyEvaluator;

impl<Spec: MCTS<State = CountingGame, TreePolicy = UCTPolicy>> Evaluator<Spec> for MyEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &CountingGame,
		moves: &Vec<Move>,
		_: Option<SearchHandle<Spec>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], state.0)
	}
	fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
		*evaln
	}
	fn evaluate_existing_state(&self, _: &CountingGame, evaln: &i64, _: SearchHandle<Spec>) -> i64 {
		*evaln
	}
}

// --- MCTS configurations ---

#[derive(Default)]
struct CountingMCTS;

impl MCTS for CountingMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ApproxTable<Self>;

	fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
		CycleBehaviour::UseCurrentEvalWhenCycleDetected
	}
}

#[derive(Default)]
struct VirtualLossMCTS;

impl MCTS for VirtualLossMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ApproxTable<Self>;

	fn virtual_loss(&self) -> i64 {
		500
	}
	fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
		CycleBehaviour::UseCurrentEvalWhenCycleDetected
	}
}

#[derive(Default)]
struct NodeLimitMCTS;

impl MCTS for NodeLimitMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ApproxTable<Self>;

	fn node_limit(&self) -> usize {
		50
	}
	fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
		CycleBehaviour::UseCurrentEvalWhenCycleDetected
	}
}

#[derive(Default)]
struct NoTranspositionMCTS;

impl MCTS for NoTranspositionMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();
}

// AlphaGo policy configuration

struct AlphaGoEvaluator;

#[derive(Default)]
struct AlphaGoMCTS;

impl MCTS for AlphaGoMCTS {
	type State = CountingGame;
	type Eval = AlphaGoEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = AlphaGoPolicy;
	type TranspositionTable = ();
}

impl Evaluator<AlphaGoMCTS> for AlphaGoEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &CountingGame,
		moves: &Vec<Move>,
		_: Option<SearchHandle<AlphaGoMCTS>>,
	) -> (Vec<f64>, i64) {
		let n = moves.len();
		let prior = if n > 0 { 1.0 / n as f64 } else { 0.0 };
		(vec![prior; n], state.0)
	}
	fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
		*evaln
	}
	fn evaluate_existing_state(
		&self,
		_: &CountingGame,
		evaln: &i64,
		_: SearchHandle<AlphaGoMCTS>,
	) -> i64 {
		*evaln
	}
}

// AlphaGo with asymmetric priors

struct AlphaGoAsymmetricEvaluator;

#[derive(Default)]
struct AlphaGoAsymmetricMCTS;

impl MCTS for AlphaGoAsymmetricMCTS {
	type State = CountingGame;
	type Eval = AlphaGoAsymmetricEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = AlphaGoPolicy;
	type TranspositionTable = ();
}

impl Evaluator<AlphaGoAsymmetricMCTS> for AlphaGoAsymmetricEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &CountingGame,
		moves: &Vec<Move>,
		_: Option<SearchHandle<AlphaGoAsymmetricMCTS>>,
	) -> (Vec<f64>, i64) {
		// Heavily favor Sub (0.9) over Add (0.1)
		let evals = if moves.len() == 2 {
			vec![0.1, 0.9]
		} else {
			vec![]
		};
		(evals, state.0)
	}
	fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
		*evaln
	}
	fn evaluate_existing_state(
		&self,
		_: &CountingGame,
		evaln: &i64,
		_: SearchHandle<AlphaGoAsymmetricMCTS>,
	) -> i64 {
		*evaln
	}
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn make_counting_mcts() -> MCTSManager<CountingMCTS> {
	MCTSManager::new(
		CountingGame(0),
		CountingMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		ApproxTable::new(1024),
	)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_single_threaded_best_move() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(10000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
	assert!(mcts.tree().num_nodes() > 1);
}

#[test]
fn test_parallel_best_move() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n_parallel(10000, 4);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_uct_visit_distribution() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(5000);
	let mut add_visits = 0u64;
	let mut sub_visits = 0u64;
	for mov in mcts.tree().root_node().moves() {
		match mov.get_move() {
			Move::Add => add_visits = mov.visits(),
			Move::Sub => sub_visits = mov.visits(),
		}
	}
	assert!(
		add_visits > sub_visits,
		"Add should get more visits than Sub: {} vs {}",
		add_visits,
		sub_visits
	);
}

#[test]
fn test_reset_clears_state() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		NoTranspositionMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(1000);
	let nodes_before = mcts.tree().num_nodes();
	assert!(nodes_before > 1);
	let mut mcts = mcts.reset();
	assert_eq!(mcts.tree().num_nodes(), 1);
	mcts.playout_n(1000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_node_limit_stops_search() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		NodeLimitMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		ApproxTable::new(1024),
	);
	mcts.print_on_playout_error(false);
	mcts.playout_n(10000);
	assert!(mcts.tree().num_nodes() <= 50);
	assert!(mcts.best_move().is_some());
}

#[test]
fn test_virtual_loss_does_not_change_result() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		VirtualLossMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		ApproxTable::new(1024),
	);
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_principal_variation() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(10000);
	let pv = mcts.principal_variation(10);
	assert_eq!(pv, vec![Move::Add; 10]);
}

#[test]
fn test_principal_variation_states() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(10000);
	let states = mcts.principal_variation_states(5);
	assert_eq!(
		states,
		vec![
			CountingGame(0),
			CountingGame(1),
			CountingGame(2),
			CountingGame(3),
			CountingGame(4),
			CountingGame(5),
		]
	);
}

#[test]
fn test_principal_variation_info() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(5000);
	let pv_info = mcts.principal_variation_info(5);
	assert!(!pv_info.is_empty());
	for info in &pv_info {
		assert!(info.visits() > 0);
		assert_eq!(info.get_move(), &Move::Add);
	}
}

#[test]
fn test_alphago_policy_basic() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		AlphaGoMCTS,
		AlphaGoEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_alphago_policy_asymmetric_priors() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		AlphaGoAsymmetricMCTS,
		AlphaGoAsymmetricEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	// Reward signal should overcome prior bias
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_transposition_table_hits() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(10000);
	let diag = mcts.tree().diagnose();
	assert!(diag.contains("transposition table hits"));
	// With Add/Sub creating cycles, there should be transposition hits
	assert!(mcts.tree().num_nodes() < 10000);
}

#[test]
fn test_no_transposition_table() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		NoTranspositionMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
	let diag = mcts.tree().diagnose();
	assert!(diag.contains("0 transposition table hits"));
}

#[test]
fn test_terminal_state() {
	let mut mcts = MCTSManager::new(
		CountingGame(100),
		CountingMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		ApproxTable::new(1024),
	);
	mcts.playout_n(100);
	assert!(mcts.best_move().is_none());
	assert!(mcts.principal_variation(10).is_empty());
	assert_eq!(mcts.tree().num_nodes(), 1);
}

#[test]
fn test_near_terminal_state() {
	let mut mcts = MCTSManager::new(
		CountingGame(99),
		CountingMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		ApproxTable::new(1024),
	);
	mcts.playout_n(1000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_playout_parallel_async() {
	let mut mcts = make_counting_mcts();
	let search = mcts.playout_parallel_async(2);
	std::thread::sleep(Duration::from_millis(100));
	search.halt();
	assert!(mcts.tree().num_nodes() > 1);
	assert!(mcts.best_move().is_some());
}

#[test]
fn test_diagnose_output() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(1000);
	let diag = mcts.tree().diagnose();
	assert!(diag.contains("nodes"));
	assert!(diag.contains("transposition table hits"));
	assert!(diag.contains("expansion contention events"));
	assert!(diag.contains("orphaned nodes"));
}

// ---------------------------------------------------------------------------
// Phase 3: Tree re-rooting tests
// ---------------------------------------------------------------------------

#[test]
fn test_advance_root_basic() {
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);

	mcts.advance(&Move::Add).unwrap();

	// Root state should have advanced
	assert_eq!(mcts.tree().root_state(), &CountingGame(1));
	// Node count reset to 1
	assert_eq!(mcts.tree().num_nodes(), 1);
	// Can continue searching from new root
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_advance_root_preserves_subtree() {
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(10000);

	// The root's Add child should have been visited
	let add_child_had_moves = mcts
		.tree()
		.root_node()
		.moves()
		.find(|m| *m.get_move() == Move::Add)
		.and_then(|m| m.child())
		.map(|c| c.moves().count() > 0)
		.unwrap_or(false);
	assert!(add_child_had_moves, "Add child should have been expanded with children");

	mcts.advance(&Move::Add).unwrap();

	// The new root should have moves (preserved from the old child)
	let root_move_count = mcts.tree().root_node().moves().count();
	assert!(root_move_count > 0, "new root should have moves from preserved subtree");
}

#[test]
fn test_advance_root_move_not_found() {
	let mut mcts = MCTSManager::new(
		CountingGame(100),
		NoTranspositionMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(100);
	let result = mcts.advance(&Move::Add);
	assert_eq!(result, Err(AdvanceError::MoveNotFound));
}

#[test]
fn test_advance_root_child_not_expanded() {
	let mut mcts = make_no_transposition_mcts();
	// No playouts — children are not expanded
	let result = mcts.advance(&Move::Add);
	assert_eq!(result, Err(AdvanceError::ChildNotExpanded));
}

#[test]
fn test_advance_root_multiple_advances() {
	let mut mcts = make_no_transposition_mcts();

	mcts.playout_n(5000);
	mcts.advance(&Move::Add).unwrap();
	assert_eq!(mcts.tree().root_state(), &CountingGame(1));

	mcts.playout_n(5000);
	mcts.advance(&Move::Add).unwrap();
	assert_eq!(mcts.tree().root_state(), &CountingGame(2));

	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_advance_root_with_transposition_table() {
	let mut mcts = make_counting_mcts();
	mcts.playout_n(5000);
	mcts.advance(&Move::Add).unwrap();
	assert_eq!(mcts.tree().root_state(), &CountingGame(1));

	// Search continues without crashes after table was cleared
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

// ---------------------------------------------------------------------------
// Phase 4.1: Depth-limited search tests
// ---------------------------------------------------------------------------

#[derive(Default)]
struct DepthLimited3MCTS;

impl MCTS for DepthLimited3MCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn max_playout_depth(&self) -> usize {
		3
	}
}

#[derive(Default)]
struct DepthLimited0MCTS;

impl MCTS for DepthLimited0MCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn max_playout_depth(&self) -> usize {
		0
	}
}

#[test]
fn test_depth_limited_finds_correct_move() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		DepthLimited3MCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_depth_limited_shallow_tree() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		DepthLimited3MCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	// PV should be shallow: depth limit 3 creates nodes at depths 1-3,
	// plus one unexpanded selection from the leaf = at most 4
	let pv = mcts.principal_variation(10);
	assert!(pv.len() <= 4, "PV length {} too deep for depth limit 3", pv.len());
}

#[test]
fn test_depth_zero_root_only() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		DepthLimited0MCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(100);
	// Tree should never grow beyond the root
	assert_eq!(mcts.tree().num_nodes(), 1);
}

#[test]
fn test_depth_limited_parallel() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		DepthLimited3MCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n_parallel(10000, 4);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

// ---------------------------------------------------------------------------
// Phase 4.3: Statistics export tests
// ---------------------------------------------------------------------------

#[test]
fn test_root_child_stats_basic() {
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	let stats = mcts.root_child_stats();
	assert_eq!(stats.len(), 2);

	let add_stats = stats.iter().find(|s| s.mov == Move::Add).unwrap();
	let sub_stats = stats.iter().find(|s| s.mov == Move::Sub).unwrap();

	assert!(add_stats.visits > 0);
	assert!(sub_stats.visits > 0);
	assert!(add_stats.avg_reward > sub_stats.avg_reward);
	assert!(add_stats.visits > sub_stats.visits);
}

#[test]
fn test_root_child_stats_no_playouts() {
	let mcts = make_no_transposition_mcts();
	let stats = mcts.root_child_stats();
	assert_eq!(stats.len(), 2);
	for s in &stats {
		assert_eq!(s.visits, 0);
		assert_eq!(s.avg_reward, 0.0);
	}
}

#[test]
fn test_root_child_stats_terminal() {
	let mcts = MCTSManager::new(
		CountingGame(100),
		NoTranspositionMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	let stats = mcts.root_child_stats();
	assert!(stats.is_empty());
}

#[test]
fn test_root_child_stats_alphago_priors() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		AlphaGoAsymmetricMCTS,
		AlphaGoAsymmetricEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	let stats = mcts.root_child_stats();
	let add_stats = stats.iter().find(|s| s.mov == Move::Add).unwrap();
	let sub_stats = stats.iter().find(|s| s.mov == Move::Sub).unwrap();

	// Priors should be preserved
	assert!((add_stats.move_evaluation - 0.1).abs() < 1e-6);
	assert!((sub_stats.move_evaluation - 0.9).abs() < 1e-6);
	// Reward signal overcomes prior bias
	assert!(add_stats.avg_reward > sub_stats.avg_reward);
}

// ---------------------------------------------------------------------------
// Phase 4.2: Progressive widening tests
// ---------------------------------------------------------------------------

/// A game with 10 possible moves (Add1..Add10). Higher-numbered moves score higher.
/// With progressive widening, only the first N moves are visible to the tree policy.
#[derive(Clone, Debug, PartialEq)]
struct WideGame(i64);

#[derive(Clone, Debug, PartialEq)]
enum WideMove {
	M(u8), // M(0) through M(9)
}

impl GameState for WideGame {
	type Move = WideMove;
	type Player = ();
	type MoveList = Vec<WideMove>;

	fn current_player(&self) -> Self::Player {}
	fn available_moves(&self) -> Vec<WideMove> {
		if self.0 >= 100 {
			vec![]
		} else {
			// Return moves 0-9, in priority order (highest value first)
			(0..10).rev().map(|i| WideMove::M(i)).collect()
		}
	}
	fn make_move(&mut self, mov: &Self::Move) {
		match mov {
			WideMove::M(i) => self.0 += *i as i64 + 1,
		}
	}
	fn max_children(&self, visits: u64) -> usize {
		// Start with 2 children, grow by 1 per 50 visits
		2 + (visits / 50) as usize
	}
}

struct WideEvaluator;

#[derive(Default)]
struct WideMCTS;

impl MCTS for WideMCTS {
	type State = WideGame;
	type Eval = WideEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();
}

impl Evaluator<WideMCTS> for WideEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &WideGame,
		moves: &Vec<WideMove>,
		_: Option<SearchHandle<WideMCTS>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], state.0)
	}
	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
		*evaln
	}
	fn evaluate_existing_state(&self, _: &WideGame, evaln: &i64, _: SearchHandle<WideMCTS>) -> i64 {
		*evaln
	}
}

#[test]
fn test_progressive_widening_limits_children() {
	let mut mcts = MCTSManager::new(
		WideGame(0),
		WideMCTS,
		WideEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	// Few playouts — widening formula is 2 + visits/50, so at ~50 root visits
	// we should see at most 3 children expanded
	mcts.playout_n(50);
	let stats = mcts.root_child_stats();
	let visited_count = stats.iter().filter(|s| s.visits > 0).count();
	assert!(
		visited_count <= 4,
		"Expected at most ~3-4 visited moves with progressive widening, got {}",
		visited_count
	);
}

#[test]
fn test_progressive_widening_expands_with_visits() {
	let mut mcts = MCTSManager::new(
		WideGame(0),
		WideMCTS,
		WideEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(100);
	let early_visited = mcts
		.root_child_stats()
		.iter()
		.filter(|s| s.visits > 0)
		.count();

	mcts.playout_n(5000);
	let late_visited = mcts
		.root_child_stats()
		.iter()
		.filter(|s| s.visits > 0)
		.count();

	assert!(
		late_visited >= early_visited,
		"More visits should expand more children: {} vs {}",
		late_visited,
		early_visited
	);
}

#[test]
fn test_progressive_widening_default_no_effect() {
	// CountingGame has default max_children (usize::MAX) — all moves visible
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	let stats = mcts.root_child_stats();
	// Both Add and Sub should have visits
	assert!(stats.iter().all(|s| s.visits > 0));
}

#[test]
fn test_progressive_widening_with_advance_root() {
	let mut mcts = MCTSManager::new(
		WideGame(0),
		WideMCTS,
		WideEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(5000);
	let best = mcts.best_move().unwrap();
	mcts.advance(&best).unwrap();
	// Continue searching from new root
	mcts.playout_n(5000);
	assert!(mcts.best_move().is_some());
}

// ---------------------------------------------------------------------------
// Seeded RNG tests
// ---------------------------------------------------------------------------

#[derive(Default)]
struct SeededMCTS;

impl MCTS for SeededMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

#[test]
fn test_seeded_rng_deterministic() {
	let run = || {
		let mut mcts = MCTSManager::new(
			CountingGame(0),
			SeededMCTS,
			MyEvaluator,
			UCTPolicy::new(0.5),
			(),
		);
		mcts.playout_n(1000);
		let stats = mcts.root_child_stats();
		(
			stats.iter().find(|s| s.mov == Move::Add).unwrap().visits,
			stats.iter().find(|s| s.mov == Move::Sub).unwrap().visits,
		)
	};

	let (add1, sub1) = run();
	let (add2, sub2) = run();
	assert_eq!(add1, add2, "Seeded search should be deterministic");
	assert_eq!(sub1, sub2, "Seeded search should be deterministic");
}

#[test]
fn test_unseeded_rng_still_works() {
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

// ---------------------------------------------------------------------------
// FPU tests
// ---------------------------------------------------------------------------

#[derive(Default)]
struct FpuMCTS;

impl MCTS for FpuMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn fpu_value(&self) -> f64 {
		0.0
	}
	fn max_playout_depth(&self) -> usize {
		20
	}
}

#[derive(Default)]
struct AlphaGoFpuMCTS;

impl MCTS for AlphaGoFpuMCTS {
	type State = CountingGame;
	type Eval = AlphaGoEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = AlphaGoPolicy;
	type TranspositionTable = ();

	fn fpu_value(&self) -> f64 {
		0.0
	}
	fn max_playout_depth(&self) -> usize {
		20
	}
}

impl Evaluator<AlphaGoFpuMCTS> for AlphaGoEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &CountingGame,
		moves: &Vec<Move>,
		_: Option<SearchHandle<AlphaGoFpuMCTS>>,
	) -> (Vec<f64>, i64) {
		let n = moves.len();
		let prior = if n > 0 { 1.0 / n as f64 } else { 0.0 };
		(vec![prior; n], state.0)
	}
	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
		*evaln
	}
	fn evaluate_existing_state(
		&self,
		_: &CountingGame,
		evaln: &i64,
		_: SearchHandle<AlphaGoFpuMCTS>,
	) -> i64 {
		*evaln
	}
}

#[test]
fn test_fpu_finds_correct_move() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		FpuMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(20000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_fpu_alphago_finds_correct_move() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		AlphaGoFpuMCTS,
		AlphaGoEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	mcts.playout_n(20000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_fpu_default_is_infinity() {
	let mcts = NoTranspositionMCTS;
	assert!(mcts.fpu_value().is_infinite());
}

// ---------------------------------------------------------------------------
// Dirichlet noise tests
// ---------------------------------------------------------------------------

#[derive(Default)]
struct NoisyAlphaGoMCTS;

impl MCTS for NoisyAlphaGoMCTS {
	type State = CountingGame;
	type Eval = AlphaGoEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = AlphaGoPolicy;
	type TranspositionTable = ();

	fn dirichlet_noise(&self) -> Option<(f64, f64)> {
		Some((0.25, 0.3))
	}
	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

impl Evaluator<NoisyAlphaGoMCTS> for AlphaGoEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &CountingGame,
		moves: &Vec<Move>,
		_: Option<SearchHandle<NoisyAlphaGoMCTS>>,
	) -> (Vec<f64>, i64) {
		let n = moves.len();
		let prior = if n > 0 { 1.0 / n as f64 } else { 0.0 };
		(vec![prior; n], state.0)
	}
	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
		*evaln
	}
	fn evaluate_existing_state(
		&self,
		_: &CountingGame,
		evaln: &i64,
		_: SearchHandle<NoisyAlphaGoMCTS>,
	) -> i64 {
		*evaln
	}
}

#[test]
fn test_dirichlet_noise_changes_priors() {
	let mcts = MCTSManager::new(
		CountingGame(0),
		NoisyAlphaGoMCTS,
		AlphaGoEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	let stats = mcts.root_child_stats();
	// Original priors are 0.5/0.5. Noise should shift them.
	let first = stats[0].move_evaluation;
	assert!(
		(first - 0.5).abs() > 1e-6,
		"Dirichlet noise should change priors, got {}",
		first
	);
}

#[test]
fn test_dirichlet_noise_deterministic_with_seed() {
	let get_priors = || {
		let mcts = MCTSManager::new(
			CountingGame(0),
			NoisyAlphaGoMCTS,
			AlphaGoEvaluator,
			AlphaGoPolicy::new(0.5),
			(),
		);
		let stats = mcts.root_child_stats();
		(stats[0].move_evaluation, stats[1].move_evaluation)
	};
	let (a1, b1) = get_priors();
	let (a2, b2) = get_priors();
	assert_eq!(a1, a2, "Seeded Dirichlet noise should be deterministic");
	assert_eq!(b1, b2, "Seeded Dirichlet noise should be deterministic");
}

#[test]
fn test_dirichlet_noise_sums_to_one() {
	let mcts = MCTSManager::new(
		CountingGame(0),
		NoisyAlphaGoMCTS,
		AlphaGoEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	let stats = mcts.root_child_stats();
	let sum: f64 = stats.iter().map(|s| s.move_evaluation).sum();
	assert!(
		(sum - 1.0).abs() < 0.01,
		"Noisy priors should sum to ~1.0, got {}",
		sum
	);
}

#[test]
fn test_dirichlet_noise_noop_for_uct() {
	// UCT uses MoveEvaluation = (), noise should be a no-op
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		SeededMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_dirichlet_noise_still_finds_correct_move() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		NoisyAlphaGoMCTS,
		AlphaGoEvaluator,
		AlphaGoPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

// ---------------------------------------------------------------------------
// Temperature selection tests
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TemperatureMCTS;

impl MCTS for TemperatureMCTS {
	type State = CountingGame;
	type Eval = MyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn selection_temperature(&self) -> f64 {
		1.0
	}
	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

#[test]
fn test_temperature_zero_is_argmax() {
	// Default temperature (0.0) should always pick most-visited = Add
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_temperature_one_mostly_picks_best() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		TemperatureMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	// With temperature 1.0, selection is proportional to visits.
	// Add has way more visits, so it should be picked most of the time.
	let mut add_count = 0;
	for _ in 0..100 {
		if mcts.best_move().unwrap() == Move::Add {
			add_count += 1;
		}
	}
	assert!(
		add_count > 80,
		"Temperature 1.0 should mostly pick Add (got {}/100)",
		add_count
	);
}

#[test]
fn test_temperature_does_not_affect_pv() {
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		TemperatureMCTS,
		MyEvaluator,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(10000);
	// PV always uses argmax regardless of temperature
	assert_eq!(mcts.principal_variation(1), vec![Move::Add]);
}

#[test]
fn test_temperature_deterministic_with_seed() {
	let run = || {
		let mut mcts = MCTSManager::new(
			CountingGame(0),
			TemperatureMCTS,
			MyEvaluator,
			UCTPolicy::new(0.5),
			(),
		);
		mcts.playout_n(5000);
		(0..10).map(|_| mcts.best_move().unwrap()).collect::<Vec<_>>()
	};
	let seq1 = run();
	let seq2 = run();
	assert_eq!(seq1, seq2, "Seeded temperature selection should be deterministic");
}

// ---------------------------------------------------------------------------
// Batched neural network evaluation tests
// ---------------------------------------------------------------------------

use std::sync::{Arc, Mutex};

struct MockBatchEvaluator {
	batch_sizes: Arc<Mutex<Vec<usize>>>,
	latency: Option<Duration>,
}

impl MockBatchEvaluator {
	fn new(batch_sizes: Arc<Mutex<Vec<usize>>>) -> Self {
		Self {
			batch_sizes,
			latency: None,
		}
	}

	fn with_latency(batch_sizes: Arc<Mutex<Vec<usize>>>, latency: Duration) -> Self {
		Self {
			batch_sizes,
			latency: Some(latency),
		}
	}
}

#[derive(Default)]
struct BatchedCountingMCTS;

impl MCTS for BatchedCountingMCTS {
	type State = CountingGame;
	type Eval = BatchedEvaluatorBridge<BatchedCountingMCTS, MockBatchEvaluator>;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
		CycleBehaviour::UseCurrentEvalWhenCycleDetected
	}
}

impl BatchEvaluator<BatchedCountingMCTS> for MockBatchEvaluator {
	type StateEvaluation = i64;

	fn evaluate_batch(
		&self,
		states: &[(CountingGame, Vec<Move>)],
	) -> Vec<(Vec<()>, i64)> {
		self.batch_sizes.lock().unwrap().push(states.len());
		if let Some(latency) = self.latency {
			std::thread::sleep(latency);
		}
		states
			.iter()
			.map(|(state, moves)| (vec![(); moves.len()], state.0))
			.collect()
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
		*evaln
	}
}

fn make_batched_mcts(
	batch_sizes: Arc<Mutex<Vec<usize>>>,
	batch_config: BatchConfig,
) -> MCTSManager<BatchedCountingMCTS> {
	let evaluator = MockBatchEvaluator::new(batch_sizes);
	let bridge = BatchedEvaluatorBridge::new(evaluator, batch_config);
	MCTSManager::new(
		CountingGame(0),
		BatchedCountingMCTS,
		bridge,
		UCTPolicy::new(0.5),
		(),
	)
}

#[test]
fn test_batched_basic_correctness() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig {
			max_batch_size: 8,
			max_wait: Duration::from_millis(1),
		},
	);
	mcts.playout_n_parallel(10000, 4);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_batched_single_threaded() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig {
			max_batch_size: 8,
			max_wait: Duration::from_millis(1),
		},
	);
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);

	let sizes = batch_sizes.lock().unwrap();
	assert!(!sizes.is_empty(), "evaluator should have been called");
	// Single-threaded: each playout blocks, so batches should all be size 1
	for &size in sizes.iter() {
		assert_eq!(size, 1, "single-threaded batches should be size 1");
	}
}

#[test]
fn test_batched_batch_size_verification() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let evaluator = MockBatchEvaluator::with_latency(
		Arc::clone(&batch_sizes),
		Duration::from_millis(2),
	);
	let bridge = BatchedEvaluatorBridge::new(
		evaluator,
		BatchConfig {
			max_batch_size: 8,
			max_wait: Duration::from_millis(5),
		},
	);
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		BatchedCountingMCTS,
		bridge,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n_parallel(1000, 4);

	let sizes = batch_sizes.lock().unwrap();
	assert!(!sizes.is_empty(), "evaluator should have been called");

	let max_batch = *sizes.iter().max().unwrap();
	assert!(
		max_batch > 1,
		"With 4 threads and latency, at least one batch should be > 1, max was {}",
		max_batch
	);

	for &size in sizes.iter() {
		assert!(
			size <= 8,
			"Batch size {} exceeds configured maximum of 8",
			size
		);
	}
}

#[test]
fn test_batched_batch_size_one() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig {
			max_batch_size: 1,
			max_wait: Duration::from_millis(1),
		},
	);
	mcts.playout_n_parallel(5000, 2);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);

	let sizes = batch_sizes.lock().unwrap();
	for &size in sizes.iter() {
		assert_eq!(size, 1, "batch_size=1 should produce only single-element batches");
	}
}

#[test]
fn test_batched_terminal_state() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let evaluator = MockBatchEvaluator::new(Arc::clone(&batch_sizes));
	let bridge = BatchedEvaluatorBridge::new(evaluator, BatchConfig::default());
	let mut mcts = MCTSManager::new(
		CountingGame(100),
		BatchedCountingMCTS,
		bridge,
		UCTPolicy::new(0.5),
		(),
	);
	mcts.playout_n(100);
	assert!(mcts.best_move().is_none());
	assert_eq!(mcts.tree().num_nodes(), 1);
}

#[test]
fn test_batched_few_playouts() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig {
			max_batch_size: 32,
			max_wait: Duration::from_millis(1),
		},
	);
	// Only 3 playouts with batch_size=32 — must flush partial batch
	mcts.playout_n_parallel(3, 2);
	assert!(mcts.best_move().is_some());

	let sizes = batch_sizes.lock().unwrap();
	assert!(!sizes.is_empty(), "partial batches should still be evaluated");
}

#[test]
fn test_batched_multi_threaded() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig {
			max_batch_size: 16,
			max_wait: Duration::from_millis(1),
		},
	);
	mcts.playout_n_parallel(10000, 8);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
	assert!(mcts.tree().num_nodes() > 1);
}

#[test]
fn test_batched_visit_distribution() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig::default(),
	);
	mcts.playout_n_parallel(5000, 4);

	let mut add_visits = 0u64;
	let mut sub_visits = 0u64;
	for mov in mcts.tree().root_node().moves() {
		match mov.get_move() {
			Move::Add => add_visits = mov.visits(),
			Move::Sub => sub_visits = mov.visits(),
		}
	}
	assert!(
		add_visits > sub_visits,
		"Add should get more visits than Sub in batched mode: {} vs {}",
		add_visits, sub_visits
	);
}

#[test]
fn test_batched_total_evaluations() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let mut mcts = make_batched_mcts(
		Arc::clone(&batch_sizes),
		BatchConfig::default(),
	);
	mcts.playout_n(1000);

	let sizes = batch_sizes.lock().unwrap();
	let total_evaluated: usize = sizes.iter().sum();
	// Each new node creation triggers one batch entry.
	// Root is created during MCTSManager::new (also through the bridge).
	let nodes_created = mcts.tree().num_nodes();
	// Total evaluations should match nodes created (including root)
	assert_eq!(
		total_evaluated, nodes_created,
		"total batch evaluations ({}) should equal nodes created ({})",
		total_evaluated, nodes_created
	);
}

// Batched AlphaGo policy tests

struct MockBatchAlphaGoEvaluator {
	batch_sizes: Arc<Mutex<Vec<usize>>>,
}

#[derive(Default)]
struct BatchedAlphaGoMCTS;

impl MCTS for BatchedAlphaGoMCTS {
	type State = CountingGame;
	type Eval = BatchedEvaluatorBridge<BatchedAlphaGoMCTS, MockBatchAlphaGoEvaluator>;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = AlphaGoPolicy;
	type TranspositionTable = ();
}

impl BatchEvaluator<BatchedAlphaGoMCTS> for MockBatchAlphaGoEvaluator {
	type StateEvaluation = i64;

	fn evaluate_batch(
		&self,
		states: &[(CountingGame, Vec<Move>)],
	) -> Vec<(Vec<f64>, i64)> {
		self.batch_sizes.lock().unwrap().push(states.len());
		states
			.iter()
			.map(|(state, moves)| {
				let n = moves.len();
				let prior = if n > 0 { 1.0 / n as f64 } else { 0.0 };
				(vec![prior; n], state.0)
			})
			.collect()
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
		*evaln
	}
}

#[test]
fn test_batched_alphago_policy() {
	let batch_sizes = Arc::new(Mutex::new(Vec::new()));
	let evaluator = MockBatchAlphaGoEvaluator {
		batch_sizes: Arc::clone(&batch_sizes),
	};
	let bridge = BatchedEvaluatorBridge::new(evaluator, BatchConfig::default());
	let mut mcts = MCTSManager::new(
		CountingGame(0),
		BatchedAlphaGoMCTS,
		bridge,
		AlphaGoPolicy::new(0.5),
		(),
	);
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

// ---------------------------------------------------------------------------
// MCTS-Solver tests
// ---------------------------------------------------------------------------

// TinyNim: two-player game for solver testing.
// Single pile of stones. Players alternate removing 1 or 2 stones.
// The player who takes the last stone(s) wins.
// Game-theoretic solution: position is losing iff stones % 3 == 0.

#[derive(Clone, Debug, PartialEq)]
struct TinyNim {
	stones: u8,
	current_player: NimPlayer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NimPlayer {
	P1,
	P2,
}

#[derive(Clone, Debug, PartialEq)]
enum NimMove {
	Take1,
	Take2,
}

impl GameState for TinyNim {
	type Move = NimMove;
	type Player = NimPlayer;
	type MoveList = Vec<NimMove>;

	fn current_player(&self) -> NimPlayer {
		self.current_player
	}

	fn available_moves(&self) -> Vec<NimMove> {
		if self.stones == 0 {
			vec![]
		} else if self.stones == 1 {
			vec![NimMove::Take1]
		} else {
			vec![NimMove::Take1, NimMove::Take2]
		}
	}

	fn make_move(&mut self, mov: &NimMove) {
		match mov {
			NimMove::Take1 => self.stones -= 1,
			NimMove::Take2 => self.stones -= 2,
		}
		self.current_player = match self.current_player {
			NimPlayer::P1 => NimPlayer::P2,
			NimPlayer::P2 => NimPlayer::P1,
		};
	}

	fn terminal_value(&self) -> Option<ProvenValue> {
		if self.stones == 0 {
			// The previous player took the last stone and won.
			// The current player (who would move next) has lost.
			Some(ProvenValue::Loss)
		} else {
			None
		}
	}
}

struct NimEvaluator;

impl<Spec: MCTS<State = TinyNim, TreePolicy = UCTPolicy>> Evaluator<Spec> for NimEvaluator {
	type StateEvaluation = Option<NimPlayer>;

	fn evaluate_new_state(
		&self,
		state: &TinyNim,
		moves: &Vec<NimMove>,
		_: Option<SearchHandle<Spec>>,
	) -> (Vec<()>, Option<NimPlayer>) {
		let winner = if state.stones == 0 {
			// Previous player won
			Some(match state.current_player {
				NimPlayer::P1 => NimPlayer::P2,
				NimPlayer::P2 => NimPlayer::P1,
			})
		} else {
			None
		};
		(vec![(); moves.len()], winner)
	}

	fn interpret_evaluation_for_player(
		&self,
		winner: &Option<NimPlayer>,
		player: &NimPlayer,
	) -> i64 {
		match winner {
			Some(w) if w == player => 100,
			Some(_) => -100,
			None => 0,
		}
	}

	fn evaluate_existing_state(
		&self,
		_: &TinyNim,
		evaln: &Option<NimPlayer>,
		_: SearchHandle<Spec>,
	) -> Option<NimPlayer> {
		*evaln
	}
}

#[derive(Default)]
struct NimSolverMCTS;

impl MCTS for NimSolverMCTS {
	type State = TinyNim;
	type Eval = NimEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn solver_enabled(&self) -> bool {
		true
	}
	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

#[derive(Default)]
struct NimNoSolverMCTS;

impl MCTS for NimNoSolverMCTS {
	type State = TinyNim;
	type Eval = NimEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

fn make_nim_solver(stones: u8) -> MCTSManager<NimSolverMCTS> {
	MCTSManager::new(
		TinyNim {
			stones,
			current_player: NimPlayer::P1,
		},
		NimSolverMCTS,
		NimEvaluator,
		UCTPolicy::new(1.0),
		(),
	)
}

#[test]
fn test_solver_nim_trivial_win_stones_1() {
	let mut mcts = make_nim_solver(1);
	mcts.playout_n(10);
	assert_eq!(mcts.best_move().unwrap(), NimMove::Take1);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
}

#[test]
fn test_solver_nim_forced_win_stones_4() {
	let mut mcts = make_nim_solver(4);
	mcts.playout_n(200);
	assert_eq!(mcts.best_move().unwrap(), NimMove::Take1);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
}

#[test]
fn test_solver_nim_forced_win_stones_5() {
	let mut mcts = make_nim_solver(5);
	mcts.playout_n(200);
	assert_eq!(mcts.best_move().unwrap(), NimMove::Take2);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
}

#[test]
fn test_solver_nim_forced_loss_stones_3() {
	let mut mcts = make_nim_solver(3);
	mcts.playout_n(200);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Loss);
}

#[test]
fn test_solver_nim_forced_loss_stones_6() {
	let mut mcts = make_nim_solver(6);
	mcts.playout_n(500);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Loss);
}

#[test]
fn test_solver_nim_all_positions_correct() {
	for stones in 1u8..=6 {
		let mut mcts = make_nim_solver(stones);
		mcts.playout_n(500);
		let expected = if stones % 3 == 0 {
			ProvenValue::Loss
		} else {
			ProvenValue::Win
		};
		assert_eq!(
			mcts.root_proven_value(),
			expected,
			"stones={}: expected {:?}, got {:?}",
			stones,
			expected,
			mcts.root_proven_value()
		);
	}
}

#[test]
fn test_solver_proven_root_stops_search() {
	let mut mcts = make_nim_solver(3);
	mcts.playout_n(200);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Loss);
	let nodes_after_prove = mcts.tree().num_nodes();

	mcts.playout_n(1000);
	assert_eq!(
		mcts.tree().num_nodes(),
		nodes_after_prove,
		"Proven root should not grow the tree"
	);
}

#[test]
fn test_solver_visit_allocation() {
	let mut mcts = make_nim_solver(4);
	mcts.playout_n(200);
	let stats = mcts.root_child_stats();
	let take1 = stats.iter().find(|s| s.mov == NimMove::Take1).unwrap();
	let take2 = stats.iter().find(|s| s.mov == NimMove::Take2).unwrap();
	assert!(
		take1.visits > take2.visits * 3,
		"Solver should favor winning move: Take1={}, Take2={}",
		take1.visits,
		take2.visits
	);
}

#[test]
fn test_solver_child_stats_proven_values() {
	let mut mcts = make_nim_solver(4);
	mcts.playout_n(200);
	let stats = mcts.root_child_stats();
	let take1 = stats.iter().find(|s| s.mov == NimMove::Take1).unwrap();
	// Take1 → stones=3 for P2 — proven Loss for P2 (child's perspective)
	// This child must be proven for root to be proven Win.
	assert_eq!(take1.proven_value, ProvenValue::Loss);
	// Root is proven Win because Take1 leads to opponent's loss
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
}

#[test]
fn test_solver_disabled_no_proven_values() {
	let mut mcts = MCTSManager::new(
		TinyNim {
			stones: 3,
			current_player: NimPlayer::P1,
		},
		NimNoSolverMCTS,
		NimEvaluator,
		UCTPolicy::new(1.0),
		(),
	);
	mcts.playout_n(200);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Unknown);
}

#[test]
fn test_solver_parallel_correctness() {
	let mut mcts = make_nim_solver(6);
	mcts.playout_n_parallel(1000, 4);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Loss);
}

#[test]
fn test_solver_default_disabled() {
	let mcts = NoTranspositionMCTS;
	assert!(!mcts.solver_enabled());
}

#[test]
fn test_solver_existing_tests_unaffected() {
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Unknown);
}

// ---------------------------------------------------------------------------
// MCTS-Solver Draw tests
// ---------------------------------------------------------------------------

// MicroGame: a configurable two-player game with hardcoded tree structure
// for testing ProvenValue::Draw propagation end-to-end.
//
// Game tree (states are u8 identifiers):
//
// Scenario 1 — Pure draw (root=10):
//   10 → {11, 12}   both terminal Draw
//
// Scenario 2 — Win trumps draw (root=20):
//   20 → {21, 22}   21=terminal Loss (parent wins), 22=terminal Draw
//
// Scenario 3 — Loss+draw = draw (root=30):
//   30 → {31, 32}   31=terminal Win (parent loses), 32=terminal Draw
//
// Scenario 4 — Multi-level draw propagation (root=40):
//   40 → {41, 42}
//   41 → {43, 44}   43=terminal Win (bad for 41), 44=terminal Draw
//   42 → {45, 46}   45=terminal Win (bad for 42), 46=terminal Draw
//   Both 41,42 resolve to Draw → 40 resolves to Draw

#[derive(Clone, Debug, PartialEq)]
struct MicroGame {
	state: u8,
	current_player: NimPlayer,
}

#[derive(Clone, Debug, PartialEq)]
struct MicroMove(u8); // target state

impl std::fmt::Display for MicroMove {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "→{}", self.0)
	}
}

impl GameState for MicroGame {
	type Move = MicroMove;
	type Player = NimPlayer;
	type MoveList = Vec<MicroMove>;

	fn current_player(&self) -> NimPlayer {
		self.current_player
	}

	fn available_moves(&self) -> Vec<MicroMove> {
		match self.state {
			10 => vec![MicroMove(11), MicroMove(12)],
			20 => vec![MicroMove(21), MicroMove(22)],
			30 => vec![MicroMove(31), MicroMove(32)],
			40 => vec![MicroMove(41), MicroMove(42)],
			41 => vec![MicroMove(43), MicroMove(44)],
			42 => vec![MicroMove(45), MicroMove(46)],
			_ => vec![], // terminal
		}
	}

	fn make_move(&mut self, mov: &MicroMove) {
		self.state = mov.0;
		self.current_player = match self.current_player {
			NimPlayer::P1 => NimPlayer::P2,
			NimPlayer::P2 => NimPlayer::P1,
		};
	}

	fn terminal_value(&self) -> Option<ProvenValue> {
		// Values from the perspective of current_player at terminal node
		match self.state {
			// Scenario 1: pure draw
			11 | 12 => Some(ProvenValue::Draw),
			// Scenario 2: 21=Loss (current player lost), 22=Draw
			21 => Some(ProvenValue::Loss),
			22 => Some(ProvenValue::Draw),
			// Scenario 3: 31=Win (current player won), 32=Draw
			31 => Some(ProvenValue::Win),
			32 => Some(ProvenValue::Draw),
			// Scenario 4: 43,45=Win (current player won=bad for parent), 44,46=Draw
			43 | 45 => Some(ProvenValue::Win),
			44 | 46 => Some(ProvenValue::Draw),
			_ => None,
		}
	}
}

struct MicroEvaluator;

impl Evaluator<DrawSolverMCTS> for MicroEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		_state: &MicroGame,
		moves: &Vec<MicroMove>,
		_: Option<SearchHandle<DrawSolverMCTS>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], 0)
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &NimPlayer) -> i64 {
		*evaln
	}

	fn evaluate_existing_state(
		&self,
		_: &MicroGame,
		evaln: &i64,
		_: SearchHandle<DrawSolverMCTS>,
	) -> i64 {
		*evaln
	}
}

#[derive(Default)]
struct DrawSolverMCTS;

impl MCTS for DrawSolverMCTS {
	type State = MicroGame;
	type Eval = MicroEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn solver_enabled(&self) -> bool {
		true
	}
	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

fn make_draw_solver(state: u8) -> MCTSManager<DrawSolverMCTS> {
	MCTSManager::new(
		MicroGame {
			state,
			current_player: NimPlayer::P1,
		},
		DrawSolverMCTS,
		MicroEvaluator,
		UCTPolicy::new(1.0),
		(),
	)
}

#[test]
fn test_solver_draw_pure_all_draws() {
	let mut mcts = make_draw_solver(10);
	mcts.playout_n(50);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Draw);
}

#[test]
fn test_solver_draw_win_trumps_draw() {
	let mut mcts = make_draw_solver(20);
	mcts.playout_n(50);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
	// best_move should pick the winning move (→21, which is Loss for opponent)
	let best = mcts.best_move().unwrap();
	assert_eq!(best, MicroMove(21));
}

#[test]
fn test_solver_draw_loss_plus_draw_is_draw() {
	let mut mcts = make_draw_solver(30);
	mcts.playout_n(50);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Draw);
	// best_move should pick the draw (→32), not the losing move (→31)
	let best = mcts.best_move().unwrap();
	assert_eq!(best, MicroMove(32));
}

#[test]
fn test_solver_draw_propagation_two_levels() {
	let mut mcts = make_draw_solver(40);
	mcts.playout_n(200);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Draw);
}

#[test]
fn test_solver_draw_proven_root_stops_search() {
	let mut mcts = make_draw_solver(10);
	mcts.playout_n(50);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Draw);
	let nodes_after = mcts.tree().num_nodes();

	mcts.playout_n(1000);
	assert_eq!(
		mcts.tree().num_nodes(),
		nodes_after,
		"Proven Draw root should not grow the tree"
	);
}

#[test]
fn test_solver_draw_child_stats_proven_values() {
	let mut mcts = make_draw_solver(30);
	mcts.playout_n(50);
	let stats = mcts.root_child_stats();
	let to_31 = stats.iter().find(|s| s.mov == MicroMove(31)).unwrap();
	let to_32 = stats.iter().find(|s| s.mov == MicroMove(32)).unwrap();
	// State 31 is Win from child's perspective (bad for parent)
	assert_eq!(to_31.proven_value, ProvenValue::Win);
	// State 32 is Draw
	assert_eq!(to_32.proven_value, ProvenValue::Draw);
}

#[test]
fn test_solver_draw_parallel_correctness() {
	let mut mcts = make_draw_solver(40);
	mcts.playout_n_parallel(500, 4);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Draw);
}

// ---------------------------------------------------------------------------
// Score-Bounded MCTS tests
// ---------------------------------------------------------------------------

// ScoreGame: a two-player game with hardcoded terminal scores for testing
// Score-Bounded MCTS. Scores are from current_player's perspective.
//
// Scenario A — depth 1 (root=0):
//   0 (P1) → {1, 2}
//   1 (P2): terminal, score=10  → parent sees -10
//   2 (P2): terminal, score=-5  → parent sees 5
//   P1 picks move→2 (score 5). Root bounds: [5, 5].
//
// Scenario B — depth 1, both negative (root=10):
//   10 (P1) → {11, 12}
//   11 (P2): terminal, score=3   → parent sees -3
//   12 (P2): terminal, score=7   → parent sees -7
//   P1 picks move→11 (score -3). Root bounds: [-3, -3].
//
// Scenario C — depth 2 (root=20):
//   20 (P1) → {21, 22}
//   21 (P2) → {23, 24}
//   22 (P2): terminal, score=0   → parent sees 0
//   23 (P1): terminal, score=8   → state 21 (P2) sees -8
//   24 (P1): terminal, score=-3  → state 21 (P2) sees 3
//   State 21 (P2): picks max(-8, 3) = 3. Bounds [3, 3].
//   State 20 (P1): picks max(-3, 0) = 0. Root bounds: [0, 0].

#[derive(Clone, Debug, PartialEq)]
struct ScoreGame {
	state: u8,
	current_player: NimPlayer,
}

#[derive(Clone, Debug, PartialEq)]
struct ScoreMove(u8);

impl std::fmt::Display for ScoreMove {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "→{}", self.0)
	}
}

impl GameState for ScoreGame {
	type Move = ScoreMove;
	type Player = NimPlayer;
	type MoveList = Vec<ScoreMove>;

	fn current_player(&self) -> NimPlayer {
		self.current_player
	}

	fn available_moves(&self) -> Vec<ScoreMove> {
		match self.state {
			0 => vec![ScoreMove(1), ScoreMove(2)],
			10 => vec![ScoreMove(11), ScoreMove(12)],
			20 => vec![ScoreMove(21), ScoreMove(22)],
			21 => vec![ScoreMove(23), ScoreMove(24)],
			_ => vec![], // terminal
		}
	}

	fn make_move(&mut self, mov: &ScoreMove) {
		self.state = mov.0;
		self.current_player = match self.current_player {
			NimPlayer::P1 => NimPlayer::P2,
			NimPlayer::P2 => NimPlayer::P1,
		};
	}

	fn terminal_score(&self) -> Option<i32> {
		match self.state {
			1 => Some(10),
			2 => Some(-5),
			11 => Some(3),
			12 => Some(7),
			22 => Some(0),
			23 => Some(8),
			24 => Some(-3),
			_ => None,
		}
	}
}

struct ScoreEvaluator;

impl Evaluator<ScoreBoundedMCTS> for ScoreEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		_state: &ScoreGame,
		moves: &Vec<ScoreMove>,
		_: Option<SearchHandle<ScoreBoundedMCTS>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], 0)
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &NimPlayer) -> i64 {
		*evaln
	}

	fn evaluate_existing_state(
		&self,
		_: &ScoreGame,
		evaln: &i64,
		_: SearchHandle<ScoreBoundedMCTS>,
	) -> i64 {
		*evaln
	}
}

#[derive(Default)]
struct ScoreBoundedMCTS;

impl MCTS for ScoreBoundedMCTS {
	type State = ScoreGame;
	type Eval = ScoreEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn score_bounded_enabled(&self) -> bool {
		true
	}
	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

fn make_score_bounded(state: u8) -> MCTSManager<ScoreBoundedMCTS> {
	MCTSManager::new(
		ScoreGame {
			state,
			current_player: NimPlayer::P1,
		},
		ScoreBoundedMCTS,
		ScoreEvaluator,
		UCTPolicy::new(1.0),
		(),
	)
}

#[test]
fn test_score_bounded_terminal_exact() {
	// Depth-1 tree: both children are terminals with known scores
	let mut mcts = make_score_bounded(0);
	mcts.playout_n(50);
	let bounds = mcts.root_score_bounds();
	// P1 picks child with -child_score = max(-10, 5) = 5
	assert_eq!(bounds, ScoreBounds::exact(5));
}

#[test]
fn test_score_bounded_best_move_selection() {
	let mut mcts = make_score_bounded(0);
	mcts.playout_n(50);
	// P1 should pick move→2 (score 5 from P1's view)
	let best = mcts.best_move().unwrap();
	assert_eq!(best, ScoreMove(2));
}

#[test]
fn test_score_bounded_all_negative() {
	// Both children give negative value to P1
	let mut mcts = make_score_bounded(10);
	mcts.playout_n(50);
	let bounds = mcts.root_score_bounds();
	// P1 picks max(-3, -7) = -3
	assert_eq!(bounds, ScoreBounds::exact(-3));
	assert_eq!(mcts.best_move().unwrap(), ScoreMove(11));
}

#[test]
fn test_score_bounded_two_levels() {
	let mut mcts = make_score_bounded(20);
	mcts.playout_n(200);
	let bounds = mcts.root_score_bounds();
	// Root value = 0 (P1 picks move→22)
	assert_eq!(bounds, ScoreBounds::exact(0));
	assert_eq!(mcts.best_move().unwrap(), ScoreMove(22));
}

#[test]
fn test_score_bounded_proven_root_stops_search() {
	let mut mcts = make_score_bounded(0);
	mcts.playout_n(50);
	assert!(mcts.root_score_bounds().is_proven());
	let nodes_after = mcts.tree().num_nodes();

	mcts.playout_n(1000);
	assert_eq!(
		mcts.tree().num_nodes(),
		nodes_after,
		"Proven root (converged bounds) should not grow the tree"
	);
}

#[test]
fn test_score_bounded_child_stats() {
	let mut mcts = make_score_bounded(0);
	mcts.playout_n(50);
	let stats = mcts.root_child_stats();
	let to_1 = stats.iter().find(|s| s.mov == ScoreMove(1)).unwrap();
	let to_2 = stats.iter().find(|s| s.mov == ScoreMove(2)).unwrap();
	// Child 1 has score 10 from P2 perspective
	assert_eq!(to_1.score_bounds, ScoreBounds::exact(10));
	// Child 2 has score -5 from P2 perspective
	assert_eq!(to_2.score_bounds, ScoreBounds::exact(-5));
}

#[test]
fn test_score_bounded_parallel_correctness() {
	let mut mcts = make_score_bounded(20);
	mcts.playout_n_parallel(500, 4);
	let bounds = mcts.root_score_bounds();
	assert_eq!(bounds, ScoreBounds::exact(0));
}

#[test]
fn test_score_bounded_disabled_by_default() {
	let mcts = NoTranspositionMCTS;
	assert!(!mcts.score_bounded_enabled());
}

#[test]
fn test_score_bounded_unbounded_without_feature() {
	// When score_bounded_enabled is false, bounds should stay unbounded
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(100);
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::UNBOUNDED);
}

// ---------------------------------------------------------------------------
// Terminal consistency tests (terminal_value + terminal_score interaction)
// ---------------------------------------------------------------------------

// UnifiedGame: a two-player game that implements BOTH terminal_value and
// terminal_score, for testing cross-derivation and consistency.
//
// State 0 (P1): moves → [1, 2, 3]
// State 1 (P2): terminal, score=10, value=Loss (P2 lost → consistent: score>0 from P2? No!)
//   Actually: terminal_score is from current_player's perspective.
//   P2 is current_player at state 1. If P2 lost, score should be negative.
//   So: score=-10 (P2 lost, negative from P2's view), value=Loss
// State 2 (P2): terminal, score=5, value=Win (P2 won, positive from P2's view)
// State 3 (P2): terminal, score=0, value=Draw

#[derive(Clone, Debug, PartialEq)]
struct UnifiedGame {
	state: u8,
	current_player: NimPlayer,
}

#[derive(Clone, Debug, PartialEq)]
struct UnifiedMove(u8);

impl std::fmt::Display for UnifiedMove {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "→{}", self.0)
	}
}

impl GameState for UnifiedGame {
	type Move = UnifiedMove;
	type Player = NimPlayer;
	type MoveList = Vec<UnifiedMove>;

	fn current_player(&self) -> NimPlayer {
		self.current_player
	}

	fn available_moves(&self) -> Vec<UnifiedMove> {
		match self.state {
			0 => vec![UnifiedMove(1), UnifiedMove(2), UnifiedMove(3)],
			_ => vec![],
		}
	}

	fn make_move(&mut self, mov: &UnifiedMove) {
		self.state = mov.0;
		self.current_player = match self.current_player {
			NimPlayer::P1 => NimPlayer::P2,
			NimPlayer::P2 => NimPlayer::P1,
		};
	}

	fn terminal_value(&self) -> Option<ProvenValue> {
		match self.state {
			1 => Some(ProvenValue::Loss),
			2 => Some(ProvenValue::Win),
			3 => Some(ProvenValue::Draw),
			_ => None,
		}
	}

	fn terminal_score(&self) -> Option<i32> {
		match self.state {
			1 => Some(-10),
			2 => Some(5),
			3 => Some(0),
			_ => None,
		}
	}
}

struct UnifiedEvaluator;

impl<Spec: MCTS<State = UnifiedGame, TreePolicy = UCTPolicy>> Evaluator<Spec> for UnifiedEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		_state: &UnifiedGame,
		moves: &Vec<UnifiedMove>,
		_: Option<SearchHandle<Spec>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], 0)
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &NimPlayer) -> i64 {
		*evaln
	}

	fn evaluate_existing_state(
		&self,
		_: &UnifiedGame,
		evaln: &i64,
		_: SearchHandle<Spec>,
	) -> i64 {
		*evaln
	}
}

// Both solver and score-bounded enabled
#[derive(Default)]
struct UnifiedBothMCTS;

impl MCTS for UnifiedBothMCTS {
	type State = UnifiedGame;
	type Eval = UnifiedEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn solver_enabled(&self) -> bool { true }
	fn score_bounded_enabled(&self) -> bool { true }
	fn rng_seed(&self) -> Option<u64> { Some(42) }
}

// ScoreOnlyGame: implements terminal_score but NOT terminal_value
// Tests that solver auto-derives proven values from scores.
#[derive(Clone, Debug, PartialEq)]
struct ScoreOnlyGame {
	state: u8,
	current_player: NimPlayer,
}

#[derive(Clone, Debug, PartialEq)]
struct ScoreOnlyMove(u8);

impl std::fmt::Display for ScoreOnlyMove {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "→{}", self.0)
	}
}

impl GameState for ScoreOnlyGame {
	type Move = ScoreOnlyMove;
	type Player = NimPlayer;
	type MoveList = Vec<ScoreOnlyMove>;

	fn current_player(&self) -> NimPlayer {
		self.current_player
	}

	fn available_moves(&self) -> Vec<ScoreOnlyMove> {
		match self.state {
			0 => vec![ScoreOnlyMove(1), ScoreOnlyMove(2)],
			_ => vec![],
		}
	}

	fn make_move(&mut self, mov: &ScoreOnlyMove) {
		self.state = mov.0;
		self.current_player = match self.current_player {
			NimPlayer::P1 => NimPlayer::P2,
			NimPlayer::P2 => NimPlayer::P1,
		};
	}

	// Only terminal_score, no terminal_value override
	fn terminal_score(&self) -> Option<i32> {
		match self.state {
			1 => Some(-10), // current player (P2) lost
			2 => Some(5),   // current player (P2) won
			_ => None,
		}
	}
}

struct ScoreOnlyEvaluator;

impl<Spec: MCTS<State = ScoreOnlyGame, TreePolicy = UCTPolicy>> Evaluator<Spec> for ScoreOnlyEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		_state: &ScoreOnlyGame,
		moves: &Vec<ScoreOnlyMove>,
		_: Option<SearchHandle<Spec>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], 0)
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &NimPlayer) -> i64 {
		*evaln
	}

	fn evaluate_existing_state(
		&self,
		_: &ScoreOnlyGame,
		evaln: &i64,
		_: SearchHandle<Spec>,
	) -> i64 {
		*evaln
	}
}

#[derive(Default)]
struct ScoreOnlySolverMCTS;

impl MCTS for ScoreOnlySolverMCTS {
	type State = ScoreOnlyGame;
	type Eval = ScoreOnlyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn solver_enabled(&self) -> bool { true }
	fn rng_seed(&self) -> Option<u64> { Some(42) }
}

// ValueOnlyGame: implements terminal_value but NOT terminal_score
// Tests that score-bounded auto-derives bounds from proven values.
#[derive(Clone, Debug, PartialEq)]
struct ValueOnlyGame {
	state: u8,
	current_player: NimPlayer,
}

#[derive(Clone, Debug, PartialEq)]
struct ValueOnlyMove(u8);

impl std::fmt::Display for ValueOnlyMove {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		write!(f, "→{}", self.0)
	}
}

impl GameState for ValueOnlyGame {
	type Move = ValueOnlyMove;
	type Player = NimPlayer;
	type MoveList = Vec<ValueOnlyMove>;

	fn current_player(&self) -> NimPlayer {
		self.current_player
	}

	fn available_moves(&self) -> Vec<ValueOnlyMove> {
		match self.state {
			0 => vec![ValueOnlyMove(1), ValueOnlyMove(2)],
			_ => vec![],
		}
	}

	fn make_move(&mut self, mov: &ValueOnlyMove) {
		self.state = mov.0;
		self.current_player = match self.current_player {
			NimPlayer::P1 => NimPlayer::P2,
			NimPlayer::P2 => NimPlayer::P1,
		};
	}

	// Only terminal_value, no terminal_score override
	fn terminal_value(&self) -> Option<ProvenValue> {
		match self.state {
			1 => Some(ProvenValue::Loss),
			2 => Some(ProvenValue::Win),
			_ => None,
		}
	}
}

struct ValueOnlyEvaluator;

impl Evaluator<ValueOnlyBoundsMCTS> for ValueOnlyEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		_state: &ValueOnlyGame,
		moves: &Vec<ValueOnlyMove>,
		_: Option<SearchHandle<ValueOnlyBoundsMCTS>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], 0)
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &NimPlayer) -> i64 {
		*evaln
	}

	fn evaluate_existing_state(
		&self,
		_: &ValueOnlyGame,
		evaln: &i64,
		_: SearchHandle<ValueOnlyBoundsMCTS>,
	) -> i64 {
		*evaln
	}
}

#[derive(Default)]
struct ValueOnlyBoundsMCTS;

impl MCTS for ValueOnlyBoundsMCTS {
	type State = ValueOnlyGame;
	type Eval = ValueOnlyEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn score_bounded_enabled(&self) -> bool { true }
	fn rng_seed(&self) -> Option<u64> { Some(42) }
}

#[test]
fn test_terminal_both_solver_and_bounds_consistent() {
	// UnifiedGame provides both terminal_value and terminal_score.
	// With both features enabled, both should be set correctly.
	let mut mcts = MCTSManager::new(
		UnifiedGame { state: 0, current_player: NimPlayer::P1 },
		UnifiedBothMCTS, UnifiedEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(50);
	// Child 1: Loss from P2 → parent (P1) wins
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
	// Root bounds: P1 picks best of negate(-10)=10, negate(5)=-5, negate(0)=0 → 10
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::exact(10));
}

#[test]
fn test_terminal_score_derives_proven_value() {
	// ScoreOnlyGame has terminal_score but NOT terminal_value.
	// Solver should auto-derive: score<0 → Loss, score>0 → Win.
	let mut mcts = MCTSManager::new(
		ScoreOnlyGame { state: 0, current_player: NimPlayer::P1 },
		ScoreOnlySolverMCTS, ScoreOnlyEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(50);
	// Child 1 has score=-10 → derived Loss. Parent picks it → Win.
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
	let stats = mcts.root_child_stats();
	let to_1 = stats.iter().find(|s| s.mov == ScoreOnlyMove(1)).unwrap();
	assert_eq!(to_1.proven_value, ProvenValue::Loss);
}

#[test]
fn test_terminal_value_derives_score_bounds() {
	// ValueOnlyGame has terminal_value but NOT terminal_score.
	// Score-bounded should auto-derive: Loss→-1, Win→1.
	let mut mcts = MCTSManager::new(
		ValueOnlyGame { state: 0, current_player: NimPlayer::P1 },
		ValueOnlyBoundsMCTS, ValueOnlyEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(50);
	// Child 1: Loss → score=-1. Child 2: Win → score=1.
	// Parent picks max(negate(-1), negate(1)) = max(1, -1) = 1
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::exact(1));
}

#[test]
fn test_terminal_both_child_stats() {
	let mut mcts = MCTSManager::new(
		UnifiedGame { state: 0, current_player: NimPlayer::P1 },
		UnifiedBothMCTS, UnifiedEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(50);
	let stats = mcts.root_child_stats();
	let to_1 = stats.iter().find(|s| s.mov == UnifiedMove(1)).unwrap();
	let to_2 = stats.iter().find(|s| s.mov == UnifiedMove(2)).unwrap();
	let to_3 = stats.iter().find(|s| s.mov == UnifiedMove(3)).unwrap();
	// Child proven values from child's perspective
	assert_eq!(to_1.proven_value, ProvenValue::Loss);
	assert_eq!(to_2.proven_value, ProvenValue::Win);
	assert_eq!(to_3.proven_value, ProvenValue::Draw);
	// Child score bounds from child's perspective
	assert_eq!(to_1.score_bounds, ScoreBounds::exact(-10));
	assert_eq!(to_2.score_bounds, ScoreBounds::exact(5));
	assert_eq!(to_3.score_bounds, ScoreBounds::exact(0));
}

#[test]
fn test_terminal_best_move_with_both() {
	let mut mcts = MCTSManager::new(
		UnifiedGame { state: 0, current_player: NimPlayer::P1 },
		UnifiedBothMCTS, UnifiedEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(50);
	// Solver should prefer the winning move (→1, child Loss = parent Win)
	assert_eq!(mcts.best_move().unwrap(), UnifiedMove(1));
}

// ---------------------------------------------------------------------------
// Solver + Score-Bounded integration tests
// ---------------------------------------------------------------------------

#[test]
fn test_integration_both_features_converge() {
	// UnifiedGame provides both terminal_value and terminal_score.
	// With both features enabled, both proven value and bounds should converge.
	let mut mcts = MCTSManager::new(
		UnifiedGame { state: 0, current_player: NimPlayer::P1 },
		UnifiedBothMCTS, UnifiedEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(100);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::exact(10));
}

#[test]
fn test_integration_bounds_convergence_sets_proven() {
	// ScoreOnlyGame has terminal_score but no terminal_value.
	// With both features enabled, terminal_value is auto-derived from score.
	// Bounds converge, and convergence auto-sets proven value.
	#[derive(Default)]
	struct ScoreOnlyBothMCTS;
	impl MCTS for ScoreOnlyBothMCTS {
		type State = ScoreOnlyGame;
		type Eval = ScoreOnlyEvaluator;
		type NodeData = ();
		type ExtraThreadData = ();
		type TreePolicy = UCTPolicy;
		type TranspositionTable = ();
		fn solver_enabled(&self) -> bool { true }
		fn score_bounded_enabled(&self) -> bool { true }
		fn rng_seed(&self) -> Option<u64> { Some(42) }
	}

	let mut mcts = MCTSManager::new(
		ScoreOnlyGame { state: 0, current_player: NimPlayer::P1 },
		ScoreOnlyBothMCTS, ScoreOnlyEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(50);
	// Solver derives proven values from terminal_score. Root proven Win.
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
	// Bounds converge: max(negate(-10), negate(5)) = max(10, -5) = 10
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::exact(10));
}

#[test]
fn test_integration_pruning_correct_result() {
	// With bounds-based pruning active, the tree policy should still
	// find the correct best move. Depth-2 tree exercises pruning logic
	// during child selection.
	let mut mcts = MCTSManager::new(
		ScoreGame { state: 20, current_player: NimPlayer::P1 },
		ScoreBoundedMCTS, ScoreEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n(200);
	// Root value is 0 (P1 picks move→22). Pruning should not break this.
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::exact(0));
	assert_eq!(mcts.best_move().unwrap(), ScoreMove(22));
}

#[test]
fn test_integration_parallel_both_features() {
	let mut mcts = MCTSManager::new(
		UnifiedGame { state: 0, current_player: NimPlayer::P1 },
		UnifiedBothMCTS, UnifiedEvaluator, UCTPolicy::new(1.0), (),
	);
	mcts.playout_n_parallel(500, 4);
	assert_eq!(mcts.root_proven_value(), ProvenValue::Win);
	assert_eq!(mcts.root_score_bounds(), ScoreBounds::exact(10));
}

// ---------------------------------------------------------------------------
// Chance node tests
// ---------------------------------------------------------------------------

// DiceGame: single-player stochastic game for chance node testing.
// Player chooses Roll or Stop each turn.
// After Roll, a d3 (1-3 uniform) is added to the score.
// Game ends when score >= 10 or player Stops.
// Optimal strategy: always Roll (E[die] = 2 > 0 = Stop value gain).

#[derive(Clone, Debug, PartialEq)]
struct DiceGame {
	score: i64,
	pending_roll: bool,
	stopped: bool,
}

impl DiceGame {
	fn new() -> Self {
		Self {
			score: 0,
			pending_roll: false,
			stopped: false,
		}
	}
	fn at(score: i64) -> Self {
		Self {
			score,
			pending_roll: false,
			stopped: false,
		}
	}
}

#[derive(Clone, Debug, PartialEq)]
enum DiceMove {
	Roll,
	Stop,
	Die(u8), // chance outcome: 1, 2, or 3
}

impl GameState for DiceGame {
	type Move = DiceMove;
	type Player = ();
	type MoveList = Vec<DiceMove>;

	fn current_player(&self) -> () {}

	fn available_moves(&self) -> Vec<DiceMove> {
		if self.pending_roll || self.stopped || self.score >= 10 {
			vec![]
		} else {
			vec![DiceMove::Roll, DiceMove::Stop]
		}
	}

	fn make_move(&mut self, mov: &DiceMove) {
		match mov {
			DiceMove::Roll => {
				self.pending_roll = true;
			}
			DiceMove::Stop => {
				self.stopped = true;
			}
			DiceMove::Die(v) => {
				self.score += *v as i64;
				self.pending_roll = false;
			}
		}
	}

	fn chance_outcomes(&self) -> Option<Vec<(DiceMove, f64)>> {
		if self.pending_roll {
			Some(vec![
				(DiceMove::Die(1), 1.0 / 3.0),
				(DiceMove::Die(2), 1.0 / 3.0),
				(DiceMove::Die(3), 1.0 / 3.0),
			])
		} else {
			None
		}
	}
}

struct DiceEvaluator;

impl<Spec: MCTS<State = DiceGame, TreePolicy = UCTPolicy>> Evaluator<Spec> for DiceEvaluator {
	type StateEvaluation = i64;

	fn evaluate_new_state(
		&self,
		state: &DiceGame,
		moves: &Vec<DiceMove>,
		_: Option<SearchHandle<Spec>>,
	) -> (Vec<()>, i64) {
		(vec![(); moves.len()], state.score)
	}

	fn interpret_evaluation_for_player(&self, evaln: &i64, _: &()) -> i64 {
		*evaln
	}

	fn evaluate_existing_state(
		&self,
		state: &DiceGame,
		_evaln: &i64,
		_: SearchHandle<Spec>,
	) -> i64 {
		// For open-loop stochastic games, re-evaluate from the current
		// (post-chance) state, since different playouts through the same
		// tree node may have different chance outcomes.
		state.score
	}
}

#[derive(Default)]
struct DiceMCTS;

impl MCTS for DiceMCTS {
	type State = DiceGame;
	type Eval = DiceEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();
}

#[derive(Default)]
struct SeededDiceMCTS;

impl MCTS for SeededDiceMCTS {
	type State = DiceGame;
	type Eval = DiceEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

fn make_dice_mcts(score: i64) -> MCTSManager<DiceMCTS> {
	MCTSManager::new(
		DiceGame::at(score),
		DiceMCTS,
		DiceEvaluator,
		UCTPolicy::new(0.5),
		(),
	)
}

#[test]
fn test_chance_roll_is_optimal_from_zero() {
	let mut mcts = make_dice_mcts(0);
	mcts.playout_n(50_000);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

#[test]
fn test_chance_roll_is_optimal_from_seven() {
	let mut mcts = make_dice_mcts(7);
	mcts.playout_n(50_000);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

#[test]
fn test_chance_roll_is_optimal_from_nine() {
	// At score 9: Stop gives 9. Roll gives E[10+11+12]/3 = 11. Roll is better.
	let mut mcts = make_dice_mcts(9);
	mcts.playout_n(10_000);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

#[test]
fn test_chance_terminal_at_ten() {
	let mut mcts = make_dice_mcts(10);
	mcts.playout_n(100);
	assert!(mcts.best_move().is_none());
}

#[test]
fn test_chance_expected_value_from_nine() {
	// From score 9, Roll: all die outcomes terminate.
	// E[score] = (10+11+12)/3 = 11
	let mut mcts = make_dice_mcts(9);
	mcts.playout_n(50_000);
	let stats = mcts.root_child_stats();
	let roll = stats.iter().find(|s| s.mov == DiceMove::Roll).unwrap();
	assert!(
		(roll.avg_reward - 11.0).abs() < 0.5,
		"Roll from 9 should have avg ~11, got {}",
		roll.avg_reward
	);
}

#[test]
fn test_chance_deterministic_game_unaffected() {
	// CountingGame returns None for chance_outcomes (default).
	let mut mcts = make_no_transposition_mcts();
	mcts.playout_n(5000);
	assert_eq!(mcts.best_move().unwrap(), Move::Add);
}

#[test]
fn test_chance_seeded_deterministic() {
	let run = || {
		let mut mcts = MCTSManager::new(
			DiceGame::new(),
			SeededDiceMCTS,
			DiceEvaluator,
			UCTPolicy::new(0.5),
			(),
		);
		mcts.playout_n(5000);
		let stats = mcts.root_child_stats();
		(
			stats.iter().find(|s| s.mov == DiceMove::Roll).unwrap().visits,
			stats.iter().find(|s| s.mov == DiceMove::Stop).unwrap().visits,
		)
	};
	let (r1, s1) = run();
	let (r2, s2) = run();
	assert_eq!(r1, r2, "Seeded stochastic search should be deterministic");
	assert_eq!(s1, s2, "Seeded stochastic search should be deterministic");
}

#[test]
fn test_chance_parallel() {
	let mut mcts = make_dice_mcts(0);
	mcts.playout_n_parallel(50_000, 4);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

// ---------------------------------------------------------------------------
// Closed-loop chance node tests
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ClosedLoopDiceMCTS;

impl MCTS for ClosedLoopDiceMCTS {
	type State = DiceGame;
	type Eval = DiceEvaluator;
	type NodeData = ();
	type ExtraThreadData = ();
	type TreePolicy = UCTPolicy;
	type TranspositionTable = ();

	fn closed_loop_chance(&self) -> bool {
		true
	}
	fn rng_seed(&self) -> Option<u64> {
		Some(42)
	}
}

fn make_closed_loop_dice(score: i64) -> MCTSManager<ClosedLoopDiceMCTS> {
	MCTSManager::new(
		DiceGame::at(score),
		ClosedLoopDiceMCTS,
		DiceEvaluator,
		UCTPolicy::new(0.5),
		(),
	)
}

#[test]
fn test_closed_loop_roll_is_optimal() {
	let mut mcts = make_closed_loop_dice(0);
	mcts.playout_n(10_000);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

#[test]
fn test_closed_loop_larger_tree_than_open_loop() {
	// Closed-loop should create more nodes (chance outcomes are stored in tree)
	let mut open = MCTSManager::new(
		DiceGame::at(0), SeededDiceMCTS, DiceEvaluator, UCTPolicy::new(0.5), (),
	);
	open.playout_n(1000);

	let mut closed = make_closed_loop_dice(0);
	closed.playout_n(1000);

	assert!(
		closed.tree().num_nodes() > open.tree().num_nodes(),
		"Closed-loop should have more nodes: closed={}, open={}",
		closed.tree().num_nodes(),
		open.tree().num_nodes()
	);
}

#[test]
fn test_closed_loop_roll_optimal_from_seven() {
	let mut mcts = make_closed_loop_dice(7);
	mcts.playout_n(10_000);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

#[test]
fn test_closed_loop_parallel() {
	let mut mcts = make_closed_loop_dice(0);
	mcts.playout_n_parallel(10_000, 4);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}

#[test]
fn test_closed_loop_default_disabled() {
	let mcts = NoTranspositionMCTS;
	assert!(!mcts.closed_loop_chance());
}

#[test]
fn test_closed_loop_open_loop_still_works() {
	// Existing open-loop behavior should be unchanged
	let mut mcts = make_dice_mcts(0);
	mcts.playout_n(10_000);
	assert_eq!(mcts.best_move().unwrap(), DiceMove::Roll);
}
