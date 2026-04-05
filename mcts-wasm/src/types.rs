use serde::Serialize;

#[derive(Serialize)]
pub struct TreeNodeJS {
    pub visits: u64,
    pub avg_reward: f64,
    pub proven: Option<String>,
    pub children: Vec<TreeEdgeJS>,
}

#[derive(Serialize)]
pub struct TreeEdgeJS {
    pub mov: String,
    pub visits: u64,
    pub avg_reward: f64,
    pub prior: Option<f64>,
    pub child: Option<TreeNodeJS>,
}

#[derive(Serialize)]
pub struct ChildStatJS {
    pub mov: String,
    pub visits: u64,
    pub avg_reward: f64,
    pub prior: Option<f64>,
    pub proven: Option<String>,
}

#[derive(Serialize)]
pub struct SearchStatsJS {
    pub total_playouts: u64,
    pub total_nodes: usize,
    pub best_move: Option<String>,
    pub children: Vec<ChildStatJS>,
}

fn proven_str(pv: mcts::ProvenValue) -> Option<String> {
    match pv {
        mcts::ProvenValue::Unknown => None,
        mcts::ProvenValue::Win => Some("Win".into()),
        mcts::ProvenValue::Loss => Some("Loss".into()),
        mcts::ProvenValue::Draw => Some("Draw".into()),
    }
}

/// Walk the search tree to `max_depth`, building a serializable snapshot.
pub fn export_tree<Spec>(
    node: mcts::NodeHandle<'_, Spec>,
    max_depth: u32,
    extract_prior: &impl Fn(&mcts::MoveEvaluation<Spec>) -> Option<f64>,
) -> TreeNodeJS
where
    Spec: mcts::MCTS,
    mcts::MoveEvaluation<Spec>: std::fmt::Debug,
    mcts::Move<Spec>: std::fmt::Display,
{
    let mut total_visits: u64 = 0;
    let mut total_reward: i64 = 0;
    let mut children = Vec::new();

    for mi in node.moves() {
        let v = mi.visits();
        let r = mi.sum_rewards();
        total_visits += v;
        total_reward += r;

        let child_node = if max_depth > 0 {
            mi.child()
                .map(|ch| export_tree::<Spec>(ch, max_depth - 1, extract_prior))
        } else {
            None
        };

        children.push(TreeEdgeJS {
            mov: format!("{}", mi.get_move()),
            visits: v,
            avg_reward: if v == 0 { 0.0 } else { r as f64 / v as f64 },
            prior: extract_prior(mi.move_evaluation()),
            child: child_node,
        });
    }

    TreeNodeJS {
        visits: total_visits,
        avg_reward: if total_visits == 0 {
            0.0
        } else {
            total_reward as f64 / total_visits as f64
        },
        proven: proven_str(node.proven_value()),
        children,
    }
}

/// Build stats from root child stats, with optional prior extraction.
pub fn build_stats<Spec>(
    manager: &mcts::MCTSManager<Spec>,
    extract_prior: impl Fn(&mcts::MoveEvaluation<Spec>) -> Option<f64>,
) -> SearchStatsJS
where
    Spec: mcts::MCTS,
    Spec::ExtraThreadData: Default,
    mcts::MoveEvaluation<Spec>: Clone,
    mcts::Move<Spec>: std::fmt::Display + Clone,
{
    let stats = manager.root_child_stats();
    let total_playouts: u64 = stats.iter().map(|s| s.visits).sum();
    let children: Vec<ChildStatJS> = stats
        .iter()
        .map(|s| ChildStatJS {
            mov: format!("{}", s.mov),
            visits: s.visits,
            avg_reward: s.avg_reward,
            prior: extract_prior(&s.move_evaluation),
            proven: proven_str(s.proven_value),
        })
        .collect();

    SearchStatsJS {
        total_playouts,
        total_nodes: manager.tree().num_nodes(),
        best_move: manager.best_move().map(|m| format!("{}", m)),
        children,
    }
}
