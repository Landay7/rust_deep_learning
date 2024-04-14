use itertools::Itertools;
use ndarray::{Array1, Array4, Axis, ShapeError};
use ndarray_npy::NpzReader;
use rust_deep_learning::configuration::Config;
use rust_deep_learning::model::sequential::{ModelError, SequentialModel};
use std::fs::{self, File};
use std::time::Instant;
use thiserror::Error;

#[derive(Error, Debug)]
enum MainError {
    #[error("Can't open hdf5 file ")]
    Hdf5OpenError(#[from] hdf5::Error),
    #[error("Can't parse config error ")]
    ConfigParseError(#[from] serde_json::Error),
    #[error("Can't build model")]
    ModelBuildError(#[from] ModelError),
    #[error("Can't read file")]
    FileReadError(#[from] std::io::Error),
    #[error("Can't read file as npz")]
    NpzError(#[from] ndarray_npy::ReadNpzError),
    #[error("Can't transform the result")]
    ArrayTransformError(#[from] ShapeError),
    #[error("Can't find max position")]
    MappingError(&'static str),
}

fn main() -> Result<(), MainError> {
    let hdf5_file = hdf5::File::open("model.weights.h5")?;
    let deserialized: Config = serde_json::from_str(fs::read_to_string("config.json")?.as_str())
        .map_err(MainError::ConfigParseError)?;
    let model = SequentialModel::from_config_and_hdf5(deserialized, &hdf5_file)?;
    let mut npz = NpzReader::new(File::open("test_data.npz")?)?;
    let x: Array4<f32> = npz.by_name("X.npy")?;
    let y: Array1<i64> = npz.by_name("y.npy")?;

    let mut correct = 0;
    let now = Instant::now();
    for (index, case) in x.axis_iter(Axis(0)).enumerate() {
        // 'case' is now a view of a 3D array representing one case
        let result_arr = model.compute(case.to_owned().into_dyn())?;
        let arr_len = result_arr.len();
        let result_1d_array = result_arr.into_shape(arr_len)?;

        let argmax_index = result_1d_array
            .iter()
            .position_max_by(|x, y| x.total_cmp(y))
            .ok_or(MainError::MappingError("Failed to find argmax index"))?;
        if argmax_index as i64 == y[index] {
            correct += 1;
        }
    }
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{}", correct);

    Ok(())
}
