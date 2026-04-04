use crate::*;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Trait for evaluators that process multiple leaf nodes in a single batch.
///
/// Neural network inference is 10-100x faster when batched. Implement this
/// trait instead of `Evaluator` when your evaluation function benefits from
/// batching (e.g., GPU-based neural networks).
///
/// Use [`BatchedEvaluatorBridge::new`] to create an `Evaluator` from a
/// `BatchEvaluator`, then pass that as `type Eval` in your MCTS config.
pub trait BatchEvaluator<Spec: MCTS>: Send + Sync + 'static {
	type StateEvaluation: Sync + Send + Clone;

	/// Evaluate a batch of newly expanded leaf nodes.
	///
	/// Each entry is a `(state, moves)` pair for a leaf that needs evaluation.
	/// Returns a `Vec` of the same length, where each element is
	/// `(move_evaluations, state_evaluation)` — the same shape as
	/// `Evaluator::evaluate_new_state`.
	fn evaluate_batch(
		&self,
		states: &[(Spec::State, MoveList<Spec>)],
	) -> Vec<(Vec<MoveEvaluation<Spec>>, Self::StateEvaluation)>;

	/// Re-evaluate a node that has already been evaluated.
	/// Called synchronously (not batched) because it is typically cheap.
	/// Default: clone the existing evaluation.
	fn evaluate_existing_state(
		&self,
		_state: &Spec::State,
		existing_evaln: &Self::StateEvaluation,
	) -> Self::StateEvaluation {
		existing_evaln.clone()
	}

	/// Convert a state evaluation to a score from a specific player's perspective.
	fn interpret_evaluation_for_player(
		&self,
		evaluation: &Self::StateEvaluation,
		player: &Player<Spec>,
	) -> i64;
}

/// Configuration for batched evaluation.
pub struct BatchConfig {
	/// Maximum leaves per batch. The collector fires when this many
	/// requests have accumulated or `max_wait` elapses.
	pub max_batch_size: usize,
	/// Maximum time to wait for a full batch after the first request arrives.
	pub max_wait: Duration,
}

impl Default for BatchConfig {
	fn default() -> Self {
		Self {
			max_batch_size: 8,
			max_wait: Duration::from_millis(1),
		}
	}
}

struct EvalRequest<Spec: MCTS, SE> {
	state: Spec::State,
	moves: MoveList<Spec>,
	response: mpsc::SyncSender<(Vec<MoveEvaluation<Spec>>, SE)>,
}

/// An `Evaluator` adapter that batches `evaluate_new_state` calls through
/// a dedicated collector thread.
///
/// Search threads calling `evaluate_new_state` enqueue their leaf state
/// and block until the batch evaluator processes them.
///
/// # Example
///
/// ```ignore
/// // 1. Implement BatchEvaluator for your NN evaluator
/// impl BatchEvaluator<MyMCTS> for MyNNEvaluator { ... }
///
/// // 2. Wrap it in a bridge
/// let bridge = BatchedEvaluatorBridge::new(my_nn_eval, BatchConfig {
///     max_batch_size: 32,
///     max_wait: Duration::from_millis(5),
/// });
///
/// // 3. Use it as a normal Evaluator
/// let mcts = MCTSManager::new(state, MyMCTS, bridge, policy, table);
/// mcts.playout_n_parallel(10000, 4);
/// ```
#[allow(clippy::type_complexity)]
pub struct BatchedEvaluatorBridge<Spec: MCTS, B: BatchEvaluator<Spec>> {
	sender: Option<Mutex<mpsc::Sender<EvalRequest<Spec, B::StateEvaluation>>>>,
	batch_eval: Arc<B>,
	eval_thread: Option<JoinHandle<()>>,
}

// SAFETY: The Mutex<Sender> is Sync, Arc<B> is Sync (B: Sync), Option<JoinHandle> is Send.
// Evaluator requires Sync, which this satisfies through the Mutex wrapper.
unsafe impl<Spec: MCTS, B: BatchEvaluator<Spec>> Sync for BatchedEvaluatorBridge<Spec, B> {}

impl<Spec, B> BatchedEvaluatorBridge<Spec, B>
where
	Spec: MCTS,
	B: BatchEvaluator<Spec>,
	Spec::State: Clone,
	MoveList<Spec>: Clone + Send + 'static,
	MoveEvaluation<Spec>: Send + 'static,
	B::StateEvaluation: Send + 'static,
{
	pub fn new(batch_eval: B, config: BatchConfig) -> Self {
		let (sender, receiver) = mpsc::channel::<EvalRequest<Spec, B::StateEvaluation>>();
		let batch_eval = Arc::new(batch_eval);
		let eval_clone = Arc::clone(&batch_eval);

		let handle = std::thread::spawn(move || {
			collector_loop(&receiver, &*eval_clone, &config);
		});

		Self {
			sender: Some(Mutex::new(sender)),
			batch_eval,
			eval_thread: Some(handle),
		}
	}
}

impl<Spec, B> Evaluator<Spec> for BatchedEvaluatorBridge<Spec, B>
where
	Spec: MCTS<Eval = Self>,
	B: BatchEvaluator<Spec>,
	Spec::State: Clone,
	MoveList<Spec>: Clone + Send + 'static,
	MoveEvaluation<Spec>: Send + 'static,
	B::StateEvaluation: Send + 'static,
{
	type StateEvaluation = B::StateEvaluation;

	fn evaluate_new_state(
		&self,
		state: &Spec::State,
		moves: &MoveList<Spec>,
		_handle: Option<SearchHandle<Spec>>,
	) -> (Vec<MoveEvaluation<Spec>>, Self::StateEvaluation) {
		let (response_tx, response_rx) = mpsc::sync_channel(1);
		let request = EvalRequest {
			state: state.clone(),
			moves: moves.clone(),
			response: response_tx,
		};
		let sender = self.sender.as_ref().expect("bridge already shut down");
		sender
			.lock()
			.unwrap()
			.send(request)
			.expect("batch collector thread died");
		response_rx.recv().expect("batch collector dropped response")
	}

	fn evaluate_existing_state(
		&self,
		state: &Spec::State,
		existing_evaln: &Self::StateEvaluation,
		_handle: SearchHandle<Spec>,
	) -> Self::StateEvaluation {
		self.batch_eval.evaluate_existing_state(state, existing_evaln)
	}

	fn interpret_evaluation_for_player(
		&self,
		evaluation: &Self::StateEvaluation,
		player: &Player<Spec>,
	) -> i64 {
		self.batch_eval
			.interpret_evaluation_for_player(evaluation, player)
	}
}

impl<Spec: MCTS, B: BatchEvaluator<Spec>> Drop for BatchedEvaluatorBridge<Spec, B> {
	fn drop(&mut self) {
		// Drop the sender to close the channel, signaling the collector to exit
		self.sender.take();
		// Join the collector thread
		if let Some(handle) = self.eval_thread.take() {
			let _ = handle.join();
		}
	}
}

fn collector_loop<Spec, B>(
	receiver: &mpsc::Receiver<EvalRequest<Spec, B::StateEvaluation>>,
	eval: &B,
	config: &BatchConfig,
) where
	Spec: MCTS,
	B: BatchEvaluator<Spec>,
	Spec::State: Clone,
	MoveList<Spec>: Clone + Send,
	MoveEvaluation<Spec>: Send,
	B::StateEvaluation: Send,
{
	let mut batch_requests: Vec<EvalRequest<Spec, B::StateEvaluation>> =
		Vec::with_capacity(config.max_batch_size);

	loop {
		// Block until at least one request arrives (or channel closes)
		batch_requests.clear();
		match receiver.recv() {
			Ok(req) => batch_requests.push(req),
			Err(_) => return, // all senders dropped — search is done
		}

		// Collect more requests up to max_batch_size or max_wait
		let deadline = Instant::now() + config.max_wait;
		while batch_requests.len() < config.max_batch_size {
			let remaining = deadline.saturating_duration_since(Instant::now());
			if remaining.is_zero() {
				break;
			}
			match receiver.recv_timeout(remaining) {
				Ok(req) => batch_requests.push(req),
				Err(mpsc::RecvTimeoutError::Timeout) => break,
				Err(mpsc::RecvTimeoutError::Disconnected) => break,
			}
		}

		// Build batch input
		let batch_input: Vec<(Spec::State, MoveList<Spec>)> = batch_requests
			.iter()
			.map(|r| (r.state.clone(), r.moves.clone()))
			.collect();

		// Evaluate the batch
		let results = eval.evaluate_batch(&batch_input);
		assert_eq!(
			results.len(),
			batch_requests.len(),
			"evaluate_batch returned {} results for {} inputs",
			results.len(),
			batch_requests.len()
		);

		// Distribute results back to waiting threads
		for (request, result) in batch_requests.drain(..).zip(results.into_iter()) {
			let _ = request.response.send(result);
		}
	}
}
