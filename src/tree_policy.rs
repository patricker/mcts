use rand::{rngs::SmallRng, Rng, SeedableRng};

use super::*;
use search_tree::*;

pub trait TreePolicy<Spec: MCTS<TreePolicy = Self>>: Sync + Sized {
	type MoveEvaluation: Sync + Send;
	type ThreadLocalData: Default;

	fn choose_child<'a, MoveIter>(&self, moves: MoveIter, handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
	where
		MoveIter: Iterator<Item = &'a MoveInfo<Spec>> + Clone;
	fn validate_evaluations(&self, _evalns: &[Self::MoveEvaluation]) {}

	/// Seed the thread-local data for deterministic search.
	/// Called when the MCTS config provides an `rng_seed()`.
	fn seed_thread_data(&self, _tld: &mut Self::ThreadLocalData, _seed: u64) {}

	/// Compare two move evaluations for ordering during progressive widening.
	/// Higher-priority moves should sort first (return `Greater` for higher priority `a`).
	/// Default: `Equal` (no reordering).
	fn compare_move_evaluations(
		&self,
		_a: &Self::MoveEvaluation,
		_b: &Self::MoveEvaluation,
	) -> std::cmp::Ordering {
		std::cmp::Ordering::Equal
	}

	/// Apply Dirichlet noise to root move evaluations for self-play exploration.
	/// Default: no-op (appropriate when MoveEvaluation is not numeric).
	fn apply_dirichlet_noise(
		&self,
		_moves: &mut [MoveInfo<Spec>],
		_epsilon: f64,
		_alpha: f64,
		_rng: &mut SmallRng,
	) {
	}
}

#[derive(Clone, Debug)]
pub struct UCTPolicy {
	exploration_constant: f64,
}

impl UCTPolicy {
	pub fn new(exploration_constant: f64) -> Self {
		assert!(
			exploration_constant > 0.0,
			"exploration constant is {} (must be positive)",
			exploration_constant
		);
		Self { exploration_constant }
	}

	pub fn exploration_constant(&self) -> f64 {
		self.exploration_constant
	}
}

const RECIPROCAL_TABLE_LEN: usize = 128;

#[derive(Clone, Debug)]
pub struct AlphaGoPolicy {
	exploration_constant: f64,
	reciprocals: Vec<f64>,
}

impl AlphaGoPolicy {
	pub fn new(exploration_constant: f64) -> Self {
		assert!(
			exploration_constant > 0.0,
			"exploration constant is {} (must be positive)",
			exploration_constant
		);
		let reciprocals = (0..RECIPROCAL_TABLE_LEN)
			.map(|x| if x == 0 { 2.0 } else { 1.0 / x as f64 })
			.collect();
		Self {
			exploration_constant,
			reciprocals,
		}
	}

	pub fn exploration_constant(&self) -> f64 {
		self.exploration_constant
	}

	fn reciprocal(&self, x: usize) -> f64 {
		if x < RECIPROCAL_TABLE_LEN {
			self.reciprocals[x]
		} else {
			1.0 / x as f64
		}
	}
}

impl<Spec: MCTS<TreePolicy = Self>> TreePolicy<Spec> for UCTPolicy {
	type ThreadLocalData = PolicyRng;
	type MoveEvaluation = ();

	fn choose_child<'a, MoveIter>(&self, moves: MoveIter, mut handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
	where
		MoveIter: Iterator<Item = &'a MoveInfo<Spec>> + Clone,
	{
		let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>();
		let adjusted_total = (total_visits + 1) as f64;
		let ln_adjusted_total = adjusted_total.ln();
		let fpu = handle.mcts().fpu_value();
		let solver = handle.mcts().solver_enabled();
		handle
			.thread_data()
			.policy_data
			.select_by_key(moves, |mov| {
				if solver {
					match mov.child_proven_value() {
						ProvenValue::Loss => return f64::INFINITY,     // child's loss = parent's win
						ProvenValue::Win => return f64::NEG_INFINITY,  // child's win = parent's loss
						ProvenValue::Draw => {
							let cv = mov.visits();
							return if cv == 0 { 0.0 } else { mov.sum_rewards() as f64 / cv as f64 };
						}
						ProvenValue::Unknown => {}
					}
				}
				let sum_rewards = mov.sum_rewards();
				let child_visits = mov.visits();
				// http://mcts.ai/pubs/mcts-survey-master.pdf
				if child_visits == 0 {
					fpu
				} else {
					let explore_term = 2.0 * (ln_adjusted_total / child_visits as f64).sqrt();
					let mean_action_value = sum_rewards as f64 / child_visits as f64;
					self.exploration_constant * explore_term + mean_action_value
				}
			})
			.unwrap()
	}

	fn seed_thread_data(&self, tld: &mut PolicyRng, seed: u64) {
		*tld = PolicyRng::seeded(seed);
	}
}

impl<Spec: MCTS<TreePolicy = Self>> TreePolicy<Spec> for AlphaGoPolicy {
	type ThreadLocalData = PolicyRng;
	type MoveEvaluation = f64;

	fn choose_child<'a, MoveIter>(&self, moves: MoveIter, mut handle: SearchHandle<Spec>) -> &'a MoveInfo<Spec>
	where
		MoveIter: Iterator<Item = &'a MoveInfo<Spec>> + Clone,
	{
		let total_visits = moves.clone().map(|x| x.visits()).sum::<u64>() + 1;
		let sqrt_total_visits = (total_visits as f64).sqrt();
		let explore_coef = self.exploration_constant * sqrt_total_visits;
		let fpu = handle.mcts().fpu_value();
		let solver = handle.mcts().solver_enabled();
		handle
			.thread_data()
			.policy_data
			.select_by_key(moves, |mov| {
				if solver {
					match mov.child_proven_value() {
						ProvenValue::Loss => return f64::INFINITY,
						ProvenValue::Win => return f64::NEG_INFINITY,
						ProvenValue::Draw => {
							let cv = mov.visits();
							return if cv == 0 { 0.0 } else { mov.sum_rewards() as f64 / cv as f64 };
						}
						ProvenValue::Unknown => {}
					}
				}
				let child_visits = mov.visits();
				if child_visits == 0 && fpu.is_finite() {
					fpu
				} else {
					let sum_rewards = mov.sum_rewards() as f64;
					let policy_evaln = *mov.move_evaluation();
					(sum_rewards + explore_coef * policy_evaln) * self.reciprocal(child_visits as usize)
				}
			})
			.unwrap()
	}

	fn validate_evaluations(&self, evalns: &[f64]) {
		for &x in evalns {
			assert!(x >= -1e-6, "Move evaluation is {} (must be non-negative)", x);
		}
		if !evalns.is_empty() {
			let evaln_sum: f64 = evalns.iter().sum();
			assert!(
				(evaln_sum - 1.0).abs() < 0.1,
				"Sum of evaluations is {} (should sum to 1)",
				evaln_sum
			);
		}
	}

	fn compare_move_evaluations(&self, a: &f64, b: &f64) -> std::cmp::Ordering {
		b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
	}

	fn seed_thread_data(&self, tld: &mut PolicyRng, seed: u64) {
		*tld = PolicyRng::seeded(seed);
	}

	fn apply_dirichlet_noise(
		&self,
		moves: &mut [MoveInfo<Spec>],
		epsilon: f64,
		alpha: f64,
		rng: &mut SmallRng,
	) {
		if moves.is_empty() {
			return;
		}
		let noise = sample_dirichlet(rng, alpha, moves.len());
		for (mi, &n) in moves.iter_mut().zip(noise.iter()) {
			let prior = *mi.move_evaluation();
			mi.set_move_evaluation((1.0 - epsilon) * prior + epsilon * n);
		}
	}
}

/// Sample from Gamma(alpha, 1) using Marsaglia-Tsang with Ahrens-Dieter boost for alpha < 1.
fn sample_gamma(rng: &mut SmallRng, alpha: f64) -> f64 {
	if alpha < 1.0 {
		// Ahrens-Dieter boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
		let u: f64 = rng.gen();
		return sample_gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
	}
	// Marsaglia-Tsang method for alpha >= 1
	let d = alpha - 1.0 / 3.0;
	let c = 1.0 / (9.0 * d).sqrt();
	loop {
		let x: f64 = loop {
			let x: f64 = rng.gen::<f64>() * 2.0 - 1.0; // uniform(-1, 1)
			// Approximate normal via ratio of uniforms (Box-Muller is overkill here)
			let y: f64 = rng.gen::<f64>() * 2.0 - 1.0;
			let r2 = x * x + y * y;
			if r2 > 0.0 && r2 < 1.0 {
				break x * (-2.0 * r2.ln() / r2).sqrt();
			}
		};
		let v = (1.0 + c * x).powi(3);
		if v <= 0.0 {
			continue;
		}
		let u: f64 = rng.gen();
		if u < 1.0 - 0.0331 * (x * x) * (x * x) {
			return d * v;
		}
		if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
			return d * v;
		}
	}
}

/// Sample from symmetric Dirichlet(alpha, ..., alpha) with n components.
fn sample_dirichlet(rng: &mut SmallRng, alpha: f64, n: usize) -> Vec<f64> {
	let mut samples: Vec<f64> = (0..n).map(|_| sample_gamma(rng, alpha)).collect();
	let sum: f64 = samples.iter().sum();
	if sum > 0.0 {
		for s in &mut samples {
			*s /= sum;
		}
	} else {
		// Fallback to uniform if all gamma samples are zero
		let uniform = 1.0 / n as f64;
		samples.fill(uniform);
	}
	samples
}

#[derive(Clone)]
pub struct PolicyRng {
	rng: SmallRng,
}

impl PolicyRng {
	pub fn new() -> Self {
		Self {
			rng: SmallRng::from_rng(rand::thread_rng()).unwrap(),
		}
	}

	pub fn seeded(seed: u64) -> Self {
		Self {
			rng: SmallRng::seed_from_u64(seed),
		}
	}

	pub fn select_by_key<T, Iter, KeyFn>(&mut self, elts: Iter, mut key_fn: KeyFn) -> Option<T>
	where
		Iter: Iterator<Item = T>,
		KeyFn: FnMut(&T) -> f64,
	{
		let mut choice = None;
		let mut num_optimal: u32 = 0;
		let mut best_so_far: f64 = f64::NEG_INFINITY;
		for elt in elts {
			let score = key_fn(&elt);
			if score > best_so_far {
				choice = Some(elt);
				num_optimal = 1;
				best_so_far = score;
			} else if score == best_so_far {
				num_optimal += 1;
				if self.rng.gen_ratio(1, num_optimal) {
					choice = Some(elt);
				}
			}
		}
		choice
	}
}

impl Default for PolicyRng {
	fn default() -> Self {
		Self::new()
	}
}
