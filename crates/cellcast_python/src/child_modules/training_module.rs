use pyo3::prelude::*;

use crate::functions::training_functions;
use crate::utils::py_import_module;

/// Registration function for the "training" module and its StarDist submodule.
pub fn register_training_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let training_module = PyModule::new(parent_module.py(), "training")?;
    let stardist_2d_module = PyModule::new(parent_module.py(), "stardist_2d")?;
    py_import_module("training", &training_module)?;
    py_import_module("training.stardist_2d", &stardist_2d_module)?;
    stardist_2d_module.add_function(wrap_pyfunction!(
        training_functions::train_stardist_2d_folder,
        &stardist_2d_module
    )?)?;
    stardist_2d_module.add_function(wrap_pyfunction!(
        training_functions::predict_stardist_2d_saved,
        &stardist_2d_module
    )?)?;
    stardist_2d_module.add_function(wrap_pyfunction!(
        training_functions::load_stardist_2d_saved,
        &stardist_2d_module
    )?)?;
    stardist_2d_module.add_function(wrap_pyfunction!(
        training_functions::predict_trained_stardist_2d,
        &stardist_2d_module
    )?)?;
    stardist_2d_module.add_class::<training_functions::PyStarDist2DModel>()?;
    training_module.add_submodule(&stardist_2d_module)?;
    parent_module.add_submodule(&training_module)
}
