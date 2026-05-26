use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Add a child module to Python's sys.modules dict.
///
/// # Description
///
/// This function manually adds a given module to Python's sys.modules
/// dict. This enables imports like `import cellcast.stardist_2d as star`.
///
/// # Arguments
///
/// * `module_name` - The name of the module to add to sys.modules.
/// * `module` - The actual PyO3 module object to register.
pub fn py_import_module(module_name: &str, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let full_name = format!("cellcast.{module_name}");
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;
    modules.set_item(full_name, module)?;
    Ok(())
}
