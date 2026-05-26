use pyo3::prelude::*;

use super::child_modules::{models_module, training_module};

/// Cellcast_python's parent module.
#[pymodule(name = "cellcast")]
fn cellcast_parent_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    models_module::register_models_module(m)?;
    training_module::register_training_module(m)?;
    Ok(())
}
