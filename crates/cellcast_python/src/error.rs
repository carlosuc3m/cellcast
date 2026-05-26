use imgal::ImgalError;
use pyo3::PyErr;
use pyo3::exceptions::PyRuntimeError;

use cellcast::training::stardist_2d::StarDistTrainError;

/// Convert an ImgalError into a RuntimeError PyErr
///
/// This is a quick/easy way to map Imgal's errors that avoids having to
/// duplicate imgal_python's map_imgal_error structure. This unfortunately,
/// casts all errors as RuntimeErrors which is untrue.
pub fn imgal_error_to_pyerr(err: ImgalError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

/// Convert a StarDist training error into a Python RuntimeError.
pub fn stardist_train_error_to_pyerr(err: StarDistTrainError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}
