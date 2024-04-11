use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Optimizer {
    module: String,
    class_name: String,
    config: HashMap<String, Value>,
    registered_name: Option<String>,
}

impl Optimizer {
    #[allow(dead_code)]
    pub fn new(
        module: String,
        class_name: String,
        config: HashMap<String, Value>,
        registered_name: Option<String>,
    ) -> Self {
        Optimizer {
            module,
            class_name,
            config,
            registered_name,
        }
    }
}
