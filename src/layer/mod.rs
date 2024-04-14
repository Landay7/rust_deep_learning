pub mod activation_layer;
pub mod dense;
pub mod flatten;

pub use activation_layer::{Activation, ActivationFunction};
pub use dense::Dense;
pub use flatten::Flatten;

use crate::{Matrix, NArray, Vector};

pub type NdResult = Result<NArray, ndarray::ShapeError>; 

pub trait Layer {
    fn compute(&self, incoming: NArray) -> NdResult;

    fn weights_mut(&mut self) -> &mut Matrix {
        panic!("this layer is not trainable")
    }

    fn biases(&self) -> &Vector {
        panic!("this layer is not trainable")
    }

    fn weights(&self) -> &Matrix {
        panic!("this layer is not trainable")
    }
}
