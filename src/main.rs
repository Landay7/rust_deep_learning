


use std::fs::{self, File};
use std::time::Instant;
use ndarray::{Array1, Array4, Axis};
use ndarray_npy::NpzReader;
use rust_deep_learning::configuration::config::Config;
use rust_deep_learning::model::sequantial::SequentialModel;
use rust_deep_learning::Vector;

fn argmax(array: &Vector) -> Option<usize> {
    let mut max_index = None;
    let mut max_value = std::f32::NEG_INFINITY;

    for (index, &value) in array.indexed_iter() {
        if value > max_value {
            max_value = value;
            max_index = Some(index);
        }
    }

    max_index
}

fn main() -> std::io::Result<()> {
    let hdf5_file = hdf5::File::open("model.weights.h5").expect("Failed to open HDF5 file");
    let deserialized: Config = serde_json::from_str(fs::read_to_string("config.json")?.as_str())?;
    let model = SequentialModel::from_config_and_hdf5(deserialized, &hdf5_file).expect("Can't build the model from hdf5");
    let mut npz = NpzReader::new(File::open("test_data.npz")?).expect("Can't read npz archieve");
    let x: Array4<f32> = npz.by_name("X.npy").expect("There is no X.npy at npz file");
    let y: Array1<i64> = npz.by_name("y.npy").expect("There is no y.npy at npz file");
    let mut index: usize = 0;
    let mut correct = 0;
    let now = Instant::now();
    for case in x.axis_iter(Axis(0)) {
        // 'case' is now a view of a 3D array representing one case
        let result_arr = model.compute(case.to_owned().into_dyn()).unwrap();
        let arr_len = result_arr.len();
        let array1 = result_arr.into_shape(arr_len).expect("Can't transform the result to 1d array");

        // Find the index of the maximum element (argmax)
        let argmax_index = argmax(&array1).unwrap();
        if argmax_index as i64 == y[index] {
            correct+= 1;
        }
        index += 1;
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{}", correct / 128);

    Ok(())
}

