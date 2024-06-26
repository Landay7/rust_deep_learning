pub mod activations;
pub mod configuration;
pub mod layer;
pub mod model;

pub type Vector = ndarray::Array1<f32>;
pub type Matrix = ndarray::Array2<f32>;
pub type NArray = ndarray::ArrayD<f32>;
