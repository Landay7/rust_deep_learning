use serde::{Deserialize, Serialize};
use crate::configuration::metric::Metric;
use crate::configuration::loss::Loss;
use crate::configuration::optimizer::Optimizer;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct CompileConfig {
    optimizer: Optimizer,
    loss: Loss,
    metrics: Vec<Metric>,
}

impl CompileConfig {
    #[allow(dead_code)]
    pub fn new(optimizer: Optimizer, loss: Loss, metrics: Vec<Metric>) -> Self {
        CompileConfig {
            optimizer,
            loss,
            metrics,
        }
    }
}