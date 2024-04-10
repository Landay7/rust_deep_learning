use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use serde_json::Value;

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Config {
    pub module: String,
    pub class_name: String,
    pub config: InnerConfig,
    pub registered_name: Option<String>,
    pub build_config: Option<HashMap<String, Value>>,
    pub compile_config: CompileConfig,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct InnerConfig {
    pub name: String,
    pub layers: Vec<Layer>,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Layer {
    pub module: String,
    pub class_name: String,
    pub config: HashMap<String, Value>,
    pub registered_name: Option<String>,
    pub build_config: Option<HashMap<String, Value>>,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct CompileConfig {
    pub optimizer: Optimizer,
    pub loss: Loss,
    pub metrics: Vec<Metric>,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Optimizer {
    pub module: String,
    pub class_name: String,
    pub config: HashMap<String, Value>,
    pub registered_name: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Loss {
    pub module: String,
    pub class_name: String,
    pub config: HashMap<String, Value>,
    pub registered_name: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Metric {
    pub module: String,
    pub class_name: String,
    pub config: HashMap<String, Value>,
    pub registered_name: Option<String>,
}


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_serialization_deserialization() {
        // Sample data
        let mut input_layer_config = HashMap::new();
        input_layer_config.insert(String::from("batch_input_shape"), serde_json::json!([null, 28, 28]));
        input_layer_config.insert(String::from("dtype"), serde_json::json!("float32"));
        input_layer_config.insert(String::from("sparse"), serde_json::json!(false));
        input_layer_config.insert(String::from("ragged"), serde_json::json!(false));
        input_layer_config.insert(String::from("name"), serde_json::json!("flatten_input"));

        let mut config = HashMap::new();
        config.insert(String::from("name"), serde_json::json!("sequential"));
        config.insert(String::from("layers"), serde_json::json!([input_layer_config]));

        let mut optimizer_config = HashMap::new();
        optimizer_config.insert(String::from("name"), serde_json::json!("Adam"));

        let mut loss_config = HashMap::new();
        loss_config.insert(String::from("reduction"), serde_json::json!("auto"));
        loss_config.insert(String::from("name"), serde_json::json!("sparse_categorical_crossentropy"));

        let mut metric_config = HashMap::new();
        metric_config.insert(String::from("name"), serde_json::json!("sparse_categorical_accuracy"));
        metric_config.insert(String::from("dtype"), serde_json::json!("float32"));

        let config = Config {
            module: String::from("keras"),
            class_name: String::from("Sequential"),
            config: InnerConfig {
                name: String::from("sequential"),
                layers: vec![
                    Layer {
                        module: String::from("keras.layers"),
                        class_name: String::from("InputLayer"),
                        config: input_layer_config,
                        registered_name: None,
                        build_config: None,
                    }
                ],
            },
            registered_name: None,
            build_config: None,
            compile_config: CompileConfig {
                optimizer: Optimizer {
                    module: String::from("keras.optimizers"),
                    class_name: String::from("Adam"),
                    config: optimizer_config,
                    registered_name: None,
                },
                loss: Loss {
                    module: String::from("keras.losses"),
                    class_name: String::from("SparseCategoricalCrossentropy"),
                    config: loss_config,
                    registered_name: None,
                },
                metrics: vec![
                    Metric {
                        module: String::from("keras.metrics"),
                        class_name: String::from("SparseCategoricalAccuracy"),
                        config: metric_config,
                        registered_name: None,
                    }
                ],
            },
        };

        let serialized = serde_json::to_string(&config).unwrap();

        let deserialized: Config = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config, deserialized);
    }
}
