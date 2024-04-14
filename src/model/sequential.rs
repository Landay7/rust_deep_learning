use crate::configuration::{Config, LayerType};
use crate::layer::{Dense, Flatten, Layer};
use crate::NArray;
use thiserror::Error;

pub struct SequentialModel {
    layers: Vec<Box<dyn Layer>>,
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Can't open hdf5 file ")]
    ParsingError(#[from] serde_json::Error),
    #[error("Can't open hdf5 file ")]
    LayerParseError(#[from] hdf5::Error),
    #[error("Can't open hdf5 file ")]
    ComputationError(#[from] ndarray::ShapeError),
    #[error("Can't open hdf5 file ")]
    ConfigurationError(&'static str),
}

impl SequentialModel {
    pub fn from_config_and_hdf5(config: Config, file: &hdf5::File) -> Result<Self, ModelError> {
        let mut layers = Vec::new();
        for layer_config in config.get_layers() {
            let layer: Box<dyn Layer> = match layer_config.get_class_name() {
                LayerType::Dense => {
                    let activation =
                        serde_json::from_value(layer_config.get_property("activation").clone())?;
                    let layer_name = layer_config.get_property("name").as_str().ok_or(
                        ModelError::ConfigurationError("Failed to find layer name"),
                    )?;
                    let dense = Dense::from_hdf5(file, layer_name, activation)?;
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
            input = layer.compute(input)?;
        }
        Ok(input)
    }
}
