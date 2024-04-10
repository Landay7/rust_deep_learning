use crate::configuration::config::Config;
use crate::layers::dense::Dense;
use crate::layers::flatten::Flatten;
use crate::layers::layer_trait::Layer;
use crate::NArray;

pub struct SequentialModel {
    layers: Vec<Box<dyn Layer>>,
}

impl SequentialModel {
    pub fn from_config_and_hdf5(config: Config, file: &hdf5::File) -> Self {
        let mut layers = Vec::new();
        for layer_config in config.config.layers {
            let layer = match layer_config.class_name.as_str() {
                "Dense" => {
                    let activation = layer_config.config["activation"].as_str().map(|val| {
                        val.to_string().into()
                    });
                    Box::new(Dense::from_hdf5(file, layer_config.config["name"].as_str().unwrap(), activation)) as Box<dyn Layer>
                }
                "Flatten" => {
                    Box::new(Flatten) as Box<dyn Layer>
                }
                "InputLayer" => { continue; }
                _ => {
                    panic!("Unsupported layer type: {}", layer_config.class_name);
                }
            };
            layers.push(layer);
        }
        SequentialModel { layers }
    }

    pub fn compute(&self, mut input: NArray) -> NArray {
        for layer in &self.layers {
            input = layer.compute(input);
        }
        input
    }
}