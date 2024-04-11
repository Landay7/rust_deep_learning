use crate::configuration::{config::Config, layer:: LayerType};
use crate::layers::dense::Dense;
use crate::layers::flatten::Flatten;
use crate::layers::layer_trait::Layer;
use crate::NArray;

pub struct SequentialModel {
    layers: Vec<Box<dyn Layer>>,
}

impl SequentialModel {
    pub fn from_config_and_hdf5(config: Config, file: &hdf5::File) -> Result<Self, String> {
        let mut layers = Vec::new();
        for layer_config in config.get_layers() {
            let layer: Box<dyn Layer> = match layer_config.get_class_name() {
                LayerType::Dense => {
                    let activation = serde_json::from_value(layer_config.get_property("activation").clone()).map_err(|err| format!("Cant parse activation function {}", err))?;
                    let layer_name = layer_config.get_property("name").as_str().ok_or(format!("can't find layer name"))?;
                    let dense = Dense::from_hdf5(file, layer_name, activation).expect(format!("Can't parse dense layer {:?}", layer_config.get_class_name()).as_str());
                    Box::new(dense)
                }
                LayerType::Flatten => {
                    Box::new(Flatten)
                }
                LayerType::InputLayer => { continue; }
            };
            layers.push(layer);
        }
        Ok(SequentialModel { layers })
    }

    pub fn compute(&self, mut input: NArray) -> Result<NArray, ndarray::ShapeError> {
        for layer in &self.layers {
            input = layer.compute(input)?;
        }
        Ok(input)
    }
}