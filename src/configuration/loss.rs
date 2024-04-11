use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Loss {
    module: String,
    class_name: String,
    config: HashMap<String, Value>,
    registered_name: Option<String>,
}

impl Loss {
    #[allow(dead_code)]
    pub fn new(module: String, class_name: String, config: HashMap<String, Value>, registered_name: Option<String>) -> Self {
        Loss {
            module,
            class_name,
            config,
            registered_name,
        }
    }
}