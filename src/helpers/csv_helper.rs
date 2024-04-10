
// Helper code, might be used in the future
use std::fs::OpenOptions;
use std::path::PathBuf;
use polars::prelude::*;
use ndarray::Array2;
use std::collections::HashMap;

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let data = CsvReader::from_path(file_path)?.has_header(true).finish()?;

    let training_dataset = data.drop("y")?;
    let training_labels = data.select(["y"])?;

    return Ok((training_dataset, training_labels));
}

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap()
        .reversed_axes()
}

pub fn write_parameters_to_json_file(
    parameters: &HashMap<String, Array2<f32>>,
    file_path: PathBuf,
) {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .unwrap();

    _ = serde_json::to_writer(file, parameters);
}
