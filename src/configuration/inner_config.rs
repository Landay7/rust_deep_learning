use crate::configuration::layer::Layer;
use serde::{Deserialize, Serialize};

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
