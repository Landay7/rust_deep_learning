use crate::NArray;
const E: f32 = std::f32::consts::E;

// Define sigmoid and relu functions
fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + f32::exp(-x))
}

fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

pub fn softmax(z: NArray) -> NArray {
    let exp_arr = z.mapv(|x| f32::powf(E, x));
    let sum: f32 = exp_arr.iter().sum();
    exp_arr.mapv(|x| x / sum)
}

pub fn sigmoid(z: NArray) -> NArray {
    z.mapv(sigmoid_scalar)
}

pub fn relu(z: NArray) -> NArray {
    z.mapv(relu_scalar)
}



#[cfg(test)]
mod tests {
    use crate::{Vector, NArray};

    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn softmax() {
        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();

        let expected_output = Vector::from_vec(vec![0.09003057, 0.24472848, 0.66524094]);

        let output = super::softmax(input);

        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn sigmoid() {
        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 0.0, -1.0]).unwrap();

        let expected_output = Vector::from(vec![0.73105858, 0.5, 0.26894142]);

        let output = super::sigmoid(input);

        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }

    #[test]
    fn relu() {
        let input = NArray::from_shape_vec(ndarray::IxDyn(&[3]), vec![-1.0, 0.0, 1.0]).unwrap();

        let expected_output = Vector::from(vec![0.0, 0.0, 1.0]);

        let output = super::relu(input);

        let output_vector = Vector::from_vec(output.as_slice().unwrap().to_vec());

        for (actual, expected) in output_vector.iter().zip(expected_output.iter()) {
            assert_approx_eq!(actual, expected, 1e-6);
        }
    }
}
