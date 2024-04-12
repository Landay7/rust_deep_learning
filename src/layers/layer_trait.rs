use crate::{Matrix, NArray, Vector};

pub trait Layer {
    fn compute(&self, incoming: NArray) -> Result<NArray, ndarray::ShapeError>;

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
