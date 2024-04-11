# rust deep learning

## Installation
To build the script, you need to install the [HDF5 module](https://portal.hdfgroup.org/downloads/). 
> **_NOTE:_** Currently, the Rust HDF5 module only supports versions 8, 10, and 12.

Add the `HDF5_DIR` to the installed directory so that the module knows where to find the binaries. All other modules can be obtained fully through Cargo.

## Running the Example

To run the example and obtain the model, follow these steps:

1. Run [keras_example.ipynb](keras_example.ipynb).
    - If you're not familiar with Python, follow these instructions:
        1. Install [TensorFlow](https://www.tensorflow.org/install/pip).
        2. Run the following command:
            ```
            pip install jupyterlab tensorflow_datasets
            ```
        3. Open Jupyter Notebook from the current folder:
            ```
            cd /path/to/current/dir
            jupyter lab
            ```
        4. Run the following command to execute all cells and obtain the result.
        ![alt text](image.png)
    - Alternatively, interact with the notebook as desired.

2. After running the notebook, you will have a `model.keras` file, which is a zip archive. 
    - **Note:** The crate currently does NOT support zip extraction, so please extract the files manually to the same folder. You will obtain the required files for this example:
        - `model.weights.h5`: Contains all the weights.
        - `config.json`: Contains module configuration.
        - `test_data.npz`: Archive with test examples used for testing.

3. At the current stage, only the `Sequential` model is supported.

4. Run the module:
    ```
    cargo run
    ```