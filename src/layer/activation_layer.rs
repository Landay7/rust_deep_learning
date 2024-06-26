use crate::NArray;
use crate::{
    activations::{relu, sigmoid, softmax},
    layer::{Layer, NdResult},
};
use serde::Deserialize;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum ActivationFunction {
    #[serde(alias = "sigmoid")]
    Sigmoid,
    #[serde(alias = "relu")]
    ReLu,
    #[serde(alias = "softmax")]
    SoftMax,
    #[serde(alias = "linear")]
    Linear,
}

impl ActivationFunction {
    pub fn compute(&self, incoming: NArray) -> NArray {
        match self {
            Self::ReLu => relu(incoming),
            Self::Sigmoid => sigmoid(incoming),
            Self::SoftMax => softmax(incoming),
            Self::Linear => incoming,
        }
    }
}

pub struct Activation {
    activation_function: ActivationFunction,
}

impl Activation {
    pub fn new(activation_function: ActivationFunction) -> Self {
        Self {
            activation_function,
        }
    }
}

impl Layer for Activation {
    fn compute(&self, incoming: NArray) -> NdResult {
        Ok(self.activation_function.compute(incoming))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn relu_activation() {
        let relu_activation = Activation::new(ActivationFunction::ReLu);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
        let expected_output = Vector::from_vec(vec![0.0, 0.0, 1.0]);
        let output = relu_activation.compute(input).unwrap();
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn sigmoid_activation() {
        let sigmoid_activation = Activation::new(ActivationFunction::Sigmoid);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0, 1.0, -1.0]).unwrap();
        let expected_output = Vector::from(vec![0.5, 0.7310586, 0.26894143]);

        let output = sigmoid_activation.compute(input).unwrap();
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn softmax_activation() {
        let softmax_activation = Activation::new(ActivationFunction::SoftMax);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expected_output = Vector::from(vec![0.09003057, 0.24472848, 0.66524094]);

        let output = softmax_activation.compute(input).unwrap();
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn linear_activation() {
        let linear_activation = Activation::new(ActivationFunction::Linear);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expected_output = Vector::from(vec![1.0, 2.0, 3.0]);

        let output = linear_activation.compute(input).unwrap();
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }
}
