
use crate::{NArray, Matrix, Vector};

pub trait Layer {
    fn compute(&self, incoming: NArray) -> NArray;
    
    fn get_mut_weights(&mut self) -> &mut Matrix {
        panic!("this layer is not trainable")
    }
    
    fn get_biases(&self) -> &Vector {
        panic!("this layer is not trainable")
    }

    fn get_weights(&self) -> &Matrix {
        panic!("this layer is not trainable")
    }
}

