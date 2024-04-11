use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::configuration::inner_config::InnerConfig;
use crate::configuration::compile_config::CompileConfig;
use crate::configuration::layer::Layer;
use serde_json::Value;

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
    pub fn get_layers(&self) -> &Vec<Layer>{
        &self.config.get_layers()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;
    use crate::configuration::loss::Loss;
    use crate::configuration::metric::Metric;
    use crate::configuration::optimizer::Optimizer;
    use crate::configuration::layer::LayerType;

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
            config: InnerConfig::new(
                String::from("sequential"),
                vec![
                    Layer::new(
                        String::from("keras.layers"),
                        LayerType::InputLayer,
                        input_layer_config,
                        None,
                        None,
                    )
                ],
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
                vec![
                    Metric::new(
                        String::from("keras.metrics"),
                        String::from("SparseCategoricalAccuracy"),
                        metric_config,
                        None,
                    )
                ],
            ),
        };

        let serialized = serde_json::to_string(&config).unwrap();

        let deserialized: Config = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config, deserialized);
    }
}
