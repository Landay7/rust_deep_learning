use crate::layer::{Layer, NdResult};
use crate::{NArray, Vector};

pub struct Flatten;

impl Layer for Flatten {
    fn compute(&self, incoming: NArray) -> NdResult {
        let raw_vec = incoming.into_raw_vec();
        (Vector::from_vec(raw_vec)).into_dimensionality()
    }
}
