use crate::NArray;
const E: f32 = std::f32::consts::E;

pub fn softmax_activation(z: NArray) -> NArray {
    let exp_arr = z.mapv(|x| f32::powf(E, x));
    let sum: f32 = exp_arr.iter().sum();
    exp_arr.mapv(|x| x / sum)
}

pub fn sigmoid_activation(z: NArray) -> NArray {
    z.mapv(|x| sigmoid(x))
}

pub fn relu_activation(z: NArray) -> NArray {
    z.mapv(|x| relu(x))
}

// Define sigmoid and relu functions
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use crate::{Vector, NArray};

    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn softmax_activation() {
        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

        let expected_output = Vector::from_vec(vec![0.09003057, 0.24472848, 0.66524094]);

        let output = super::softmax_activation(input);

        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn sigmoid_activation() {
        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 0.0, -1.0]).unwrap();

        let expected_output = Vector::from(vec![0.73105858, 0.5, 0.26894142]);

        let output = super::sigmoid_activation(input);

        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn relu_activation() {
        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();

        let expected_output = Vector::from(vec![0.0, 0.0, 1.0]);

        let output = super::relu_activation(input);

        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }
}
