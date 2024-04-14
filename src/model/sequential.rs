use crate::configuration::{config::Config, layer::LayerType};
use crate::layers::dense::Dense;
use crate::layers::flatten::Flatten;
use crate::layers::Layer;
use crate::NArray;

pub struct SequentialModel {
    layers: Vec<Box<dyn Layer>>,
}

#[derive(Debug)]
pub enum ModelError {
    ParsingError(serde_json::Error),
    LayerParseError(hdf5::Error),
    ComputationError(ndarray::ShapeError),
    ConfigurationError(String),
}

impl SequentialModel {
    pub fn from_config_and_hdf5(config: Config, file: &hdf5::File) -> Result<Self, ModelError> {
        let mut layers = Vec::new();
        for layer_config in config.get_layers() {
            let layer: Box<dyn Layer> = match layer_config.get_class_name() {
                LayerType::Dense => {
                    let activation =
                        serde_json::from_value(layer_config.get_property("activation").clone())
                            .map_err(ModelError::ParsingError)?;
                    let layer_name = layer_config.get_property("name").as_str().ok_or(
                        ModelError::ConfigurationError("Failed to find layer name".to_string()),
                    )?;
                    let dense = Dense::from_hdf5(file, layer_name, activation)
                        .map_err(ModelError::LayerParseError)?;
                    Box::new(dense)
                }
                LayerType::Flatten => Box::new(Flatten),
                LayerType::InputLayer => continue,
            };
            layers.push(layer);
        }
        Ok(SequentialModel { layers })
    }

    pub fn compute(&self, mut input: NArray) -> Result<NArray, ModelError> {
        for layer in &self.layers {
            input = layer.compute(input).map_err(ModelError::ComputationError)?;
        }
        Ok(input)
    }
}
