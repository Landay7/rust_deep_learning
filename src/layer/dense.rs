use crate::layer::{ActivationFunction, Layer, NdResult};
use crate::{Matrix, NArray, Vector};

pub struct Dense {
    weights: Matrix,
    bias: Vector,
    activation: Option<ActivationFunction>,
}

impl Dense {
    pub fn new(weights: Matrix, bias: Vector, activation: Option<ActivationFunction>) -> Self {
        Self {
            weights,
            bias,
            activation,
        }
    }

    pub fn from_hdf5(
        file: &hdf5::File,
        layer_name: &str,
        activation: Option<ActivationFunction>,
    ) -> Result<Self, hdf5::Error> {
        let base_path = format!(r"/layers\{layer_name}/vars");
        dbg!(&base_path);
        let weights: Matrix = file
            .dataset(format!("{}/0", base_path).as_str())?
            .read_2d()?;
        let bias: Vector = file
            .dataset(format!("{}/1", base_path).as_str())?
            .read_1d()?;
        Ok(Self::new(weights, bias, activation))
    }
}

impl Layer for Dense {
    fn compute(&self, incoming: NArray) -> NdResult {
        let incoming_len = incoming.len();
        let arr_1d: Vector = incoming.into_shape(incoming_len)?;
        let computation_result = arr_1d.dot(&self.weights) + self.bias.clone();
        if let Some(activation) = &self.activation {
            let result_1d = computation_result.into_dimensionality()?;
            Ok(activation.compute(result_1d))
        } else {
            computation_result.into_dimensionality()
        }
    }

    fn weights(&self) -> &Matrix {
        &self.weights
    }

    fn biases(&self) -> &Vector {
        &self.bias
    }

    fn weights_mut(&mut self) -> &mut Matrix {
        &mut self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layer::ActivationFunction;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_dense_constructor() {
        let weights = Matrix::ones((3, 3));
        let bias = Vector::zeros(3);

        let dense_without_activation = Dense::new(weights.clone(), bias.clone(), None);
        assert_eq!(dense_without_activation.weights, weights);
        assert_eq!(dense_without_activation.bias, bias);
        assert_eq!(dense_without_activation.activation, None);
    }

    #[test]
    fn test_dense_compute_without_activation() {
        let weights =
            Matrix::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let bias = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let dense_layer = Dense::new(weights, bias, None);

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

        let output = dense_layer.compute(input.clone()).unwrap();
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        let expected_output = Vector::from_vec(vec![31.0, 38.0, 45.0]);
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn test_dense_compute_with_activation() {
        let weights = Matrix::from_shape_vec(
            (3, 3),
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0],
        )
        .unwrap();
        let bias = Vector::from_vec(vec![1.0, -2.0, 3.0]);
        let activation = ActivationFunction::ReLu;
        let dense_layer = Dense::new(weights, bias, Some(activation));

        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 2.0, -3.0]).unwrap();

        let output = dense_layer.compute(input.clone()).unwrap();
        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        let expected_output = Vector::from_vec(vec![0.0, 34.0, 0.0]);
        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }
}
