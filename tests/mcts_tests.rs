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
	mcts.playout_n(10000);
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
	mcts.playout_n(10000);
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
