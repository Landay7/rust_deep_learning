use crate::{activations::functions::{relu_activation, sigmoid_activation, softmax_activation}, layers::layer_trait::Layer};
use crate::NArray;

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActivationFunction {
    Sigmoid,
    ReLu,
    SoftMax,
    Linear
}

impl ActivationFunction {
    pub fn compute(&self, incoming: NArray) -> NArray {
        match self{
            ActivationFunction::ReLu => {
                relu_activation(incoming)
            },
            ActivationFunction::Sigmoid => {
                sigmoid_activation(incoming)
            },
            ActivationFunction::SoftMax => {
                softmax_activation(incoming)
            },
            ActivationFunction::Linear => {
                incoming
            }
        }
    }
}

impl Into<ActivationFunction> for String{
    fn into(self) -> ActivationFunction {
        match self.as_str() {
            "relu" => ActivationFunction::ReLu,
            "sigmoid" => ActivationFunction::Sigmoid,
            "softmax" => ActivationFunction::SoftMax,
            "linear" => ActivationFunction::Linear,
            _ => panic!("Unimplemented Activation function {}", self)
        }
    }
}

pub struct Activation {
    pub threshold: f64,
    pub activation_function: ActivationFunction
}

impl Activation{
    pub fn new(threshold: f64, activation_function: ActivationFunction) -> Self{
        Self {
            threshold,
            activation_function
        }
    }
}

impl Layer for Activation {
    fn compute(&self, incoming: NArray) -> NArray {
        self.activation_function.compute(incoming)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vector;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_relu_activation() {
        let relu_activation = Activation::new(0.0, ActivationFunction::ReLu);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();
        let expected_output = Vector::from_vec(vec![0.0, 0.0, 1.0]);
        let output = relu_activation.compute(input);
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_activation() {
        let sigmoid_activation = Activation::new(0.0, ActivationFunction::Sigmoid);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.0, 1.0, -1.0]).unwrap();
        let expected_output = Vector::from(vec![0.5, 0.7310586, 0.26894143]);
        
        let output = sigmoid_activation.compute(input);
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn test_softmax_activation() {
        let softmax_activation = Activation::new(0.0, ActivationFunction::SoftMax);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expected_output = Vector::from(vec![0.09003057, 0.24472848, 0.66524094]);
        
        let output = softmax_activation.compute(input);
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn test_linear_activation() {
        let linear_activation = Activation::new(0.0, ActivationFunction::Linear);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expected_output = Vector::from(vec![1.0, 2.0, 3.0]);
        
        let output = linear_activation.compute(input);
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }
}

