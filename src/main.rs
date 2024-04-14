use ndarray::{Array1, Array4, Axis};
use ndarray_npy::NpzReader;
use rust_deep_learning::configuration::config::Config;
use rust_deep_learning::model::sequential::SequentialModel;
use std::fs::{self, File};
use std::time::Instant;
use itertools::Itertools;

#[derive(Debug)]
enum MainError {
    Hdf5OpenError(String),
    ConfigParseError(serde_json::Error),
    ModelBuildError(String),
    FileReadError(std::io::Error),
    NpzReadError(ndarray_npy::ReadNpzError),
    NpzMissingFileError(ndarray_npy::ReadNpzError),
    ModelComputeError(String),
    ArrayTransformError(String),
}

fn main() -> Result<(), MainError> {
    let hdf5_file = hdf5::File::open("model.weights.h5")
        .map_err(|err| MainError::Hdf5OpenError(format!("Failed to open HDF5 file: {}", err)))?;
    let deserialized: Config = serde_json::from_str(
        fs::read_to_string("config.json")
            .map_err(MainError::FileReadError)?
            .as_str(),
    )
    .map_err(MainError::ConfigParseError)?;
    let model = SequentialModel::from_config_and_hdf5(deserialized, &hdf5_file).map_err(|err| {
        MainError::ModelBuildError(format!("Can't build the model from hdf5: {:?}", err))
    })?;
    let mut npz =
        NpzReader::new(File::open("test_data.npz").map_err(MainError::FileReadError)?).map_err(MainError::NpzReadError)?;
    let x: Array4<f32> = npz
        .by_name("X.npy")
        .map_err(MainError::NpzMissingFileError)?;
    let y: Array1<i64> = npz
        .by_name("y.npy")
        .map_err(MainError::NpzMissingFileError)?;

    let mut correct = 0;
    let now = Instant::now();
    for (index, case) in x.axis_iter(Axis(0)).enumerate() {
        // 'case' is now a view of a 3D array representing one case
        let result_arr = model.compute(case.to_owned().into_dyn()).map_err(|err| {
            MainError::ModelComputeError(format!("Layer computation error: {:?}", err))
        })?;
        let arr_len = result_arr.len();
        let array1 = result_arr.into_shape(arr_len).map_err(|err| {
            MainError::ArrayTransformError(format!(
                "Can't transform the result to 1d array: {}",
                err
            ))
        })?;

        // Find the index of the maximum element (argmax)
        let argmax_index = array1.iter().position_max_by(|x, y| x.total_cmp(y)).ok_or(MainError::ArrayTransformError(
            "Failed to find argmax index".to_string(),
        ))?;
        if argmax_index as i64 == y[index] {
            correct += 1;
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{}", correct);

    Ok(())
}
