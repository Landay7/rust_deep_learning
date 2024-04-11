use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum LayerType {
    Dense,
    Flatten,
    InputLayer,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Layer {
    module: String,
    class_name: LayerType,
    config: HashMap<String, Value>,
    registered_name: Option<String>,
    build_config: Option<HashMap<String, Value>>,
}

impl Layer {
    pub fn get_class_name(&self) -> &LayerType {
        &self.class_name
    }

    pub fn get_property(&self, property_name: &str) -> &Value {
        &self.config[property_name]
    }

    pub fn new(
        module: String,
        class_name: LayerType,
        config: HashMap<String, Value>,
        registered_name: Option<String>,
        build_config: Option<HashMap<String, Value>>,
    ) -> Self {
        Layer {
            module,
            class_name,
            config,
            registered_name,
            build_config,
        }
    }
}
