use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use serde_json::Value;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Metric {
    module: String,
    class_name: String,
    config: HashMap<String, Value>,
    registered_name: Option<String>,
}

impl Metric {
    #[allow(dead_code)]
    pub fn new(module: String, class_name: String, config: HashMap<String, Value>, registered_name: Option<String>) -> Self {
        Metric {
            module,
            class_name,
            config,
            registered_name,
        }
    }
}