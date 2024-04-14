use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Config {
    module: String,
    class_name: String,
    config: InnerConfig,
    registered_name: Option<String>,
    build_config: Option<HashMap<String, Value>>,
    compile_config: CompileConfig,
}

impl Config {
    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.config.get_layers()
    }
}

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

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct InnerConfig {
    name: String,
    layers: Vec<Layer>,
}

impl InnerConfig {
    pub fn get_layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    #[allow(dead_code)]
    pub fn new(name: String, layers: Vec<Layer>) -> Self {
        InnerConfig { name, layers }
    }
}

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

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Loss {
    module: String,
    class_name: String,
    config: HashMap<String, Value>,
    registered_name: Option<String>,
}

impl Loss {
    #[allow(dead_code)]
    pub fn new(
        module: String,
        class_name: String,
        config: HashMap<String, Value>,
        registered_name: Option<String>,
    ) -> Self {
        Loss {
            module,
            class_name,
            config,
            registered_name,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Metric {
    module: String,
    class_name: String,
    config: HashMap<String, Value>,
    registered_name: Option<String>,
}

impl Metric {
    #[allow(dead_code)]
    pub fn new(
        module: String,
        class_name: String,
        config: HashMap<String, Value>,
        registered_name: Option<String>,
    ) -> Self {
        Metric {
            module,
            class_name,
            config,
            registered_name,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_serialization_deserialization() {
        // Sample data
        let mut input_layer_config = HashMap::new();
        input_layer_config.insert(
            String::from("batch_input_shape"),
            serde_json::json!([null, 28, 28]),
        );
        input_layer_config.insert(String::from("dtype"), serde_json::json!("float32"));
        input_layer_config.insert(String::from("sparse"), serde_json::json!(false));
        input_layer_config.insert(String::from("ragged"), serde_json::json!(false));
        input_layer_config.insert(String::from("name"), serde_json::json!("flatten_input"));

        let mut config = HashMap::new();
        config.insert(String::from("name"), serde_json::json!("sequential"));
        config.insert(
            String::from("layers"),
            serde_json::json!([input_layer_config]),
        );

        let mut optimizer_config = HashMap::new();
        optimizer_config.insert(String::from("name"), serde_json::json!("Adam"));

        let mut loss_config = HashMap::new();
        loss_config.insert(String::from("reduction"), serde_json::json!("auto"));
        loss_config.insert(
            String::from("name"),
            serde_json::json!("sparse_categorical_crossentropy"),
        );

        let mut metric_config = HashMap::new();
        metric_config.insert(
            String::from("name"),
            serde_json::json!("sparse_categorical_accuracy"),
        );
        metric_config.insert(String::from("dtype"), serde_json::json!("float32"));

        let config = Config {
            module: String::from("keras"),
            class_name: String::from("Sequential"),
            config: InnerConfig::new(
                String::from("sequential"),
                vec![Layer::new(
                    String::from("keras.layers"),
                    LayerType::InputLayer,
                    input_layer_config,
                    None,
                    None,
                )],
            ),
            registered_name: None,
            build_config: None,
            compile_config: CompileConfig::new(
                Optimizer::new(
                    String::from("keras.optimizers"),
                    String::from("Adam"),
                    optimizer_config,
                    None,
                ),
                Loss::new(
                    String::from("keras.losses"),
                    String::from("SparseCategoricalCrossentropy"),
                    loss_config,
                    None,
                ),
                vec![Metric::new(
                    String::from("keras.metrics"),
                    String::from("SparseCategoricalAccuracy"),
                    metric_config,
                    None,
                )],
            ),
        };

        let serialized = serde_json::to_string(&config).unwrap();

        let deserialized: Config = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config, deserialized);
    }
}
