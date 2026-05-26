use std::path::PathBuf;

use cellcast::training::io_2d::{
    load_training_dataset_from_folder_with_options,
    load_training_samples_from_folders_with_options, split_train_valid_2d, FolderDatasetOptions2D,
    ImageChannels2D, LabelColorMode2D,
};
use cellcast::training::stardist_2d::{
    save_stardist_2d, train_stardist_2d_with_callbacks, AugmentConfig2D, CpuTrainBackend,
    EpochEndEvent2D, EpochMetrics, LearningRateSchedule2D, Normalization2D, StarDistTrainError,
    StepMetrics2D, TrainingCallbacks2D, TrainingConfig2D, TrainingPlan2D, TrainingResult2D,
    TrainingSample2D, ValidationPreview2D, WgpuTrainBackend,
};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::stardist_train_error_to_pyerr;

const DEFAULT_VALID_FRACTION_2D: f32 = 0.15;

/// Python-facing summary returned after training finishes.
struct PythonTrainingSummary2D {
    history: Vec<EpochMetrics>,
    config: TrainingConfig2D,
    output_dir: Option<PathBuf>,
    train_samples: usize,
    valid_samples: usize,
}

/// Bridge from Rust training events to Python callables.
struct PythonTrainingCallbacks2D {
    on_train_begin: Option<Py<PyAny>>,
    on_step_end: Option<Py<PyAny>>,
    on_validation_end: Option<Py<PyAny>>,
}

impl PythonTrainingCallbacks2D {
    fn new(
        on_train_begin: Option<Py<PyAny>>,
        on_step_end: Option<Py<PyAny>>,
        on_validation_end: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            on_train_begin,
            on_step_end,
            on_validation_end,
        }
    }
}

impl TrainingCallbacks2D for PythonTrainingCallbacks2D {
    fn on_train_begin(&mut self, plan: &TrainingPlan2D) -> Result<(), StarDistTrainError> {
        if let Some(callback) = &self.on_train_begin {
            Python::attach(|py| {
                let payload = training_plan_to_pydict(py, plan)?;
                callback.bind(py).call1((payload,))?;
                Ok(())
            })
            .map_err(|err: PyErr| StarDistTrainError::Callback(err.to_string()))?;
        }
        Ok(())
    }

    fn on_step_end(&mut self, metrics: &StepMetrics2D) -> Result<(), StarDistTrainError> {
        if let Some(callback) = &self.on_step_end {
            Python::attach(|py| {
                let payload = step_metrics_to_pydict(py, metrics)?;
                callback.bind(py).call1((payload,))?;
                Ok(())
            })
            .map_err(|err: PyErr| StarDistTrainError::Callback(err.to_string()))?;
        }
        Ok(())
    }

    fn on_epoch_end(&mut self, event: &EpochEndEvent2D) -> Result<(), StarDistTrainError> {
        if let Some(callback) = &self.on_validation_end {
            Python::attach(|py| {
                let payload = epoch_event_to_pydict(py, event)?;
                callback.bind(py).call1((payload,))?;
                Ok(())
            })
            .map_err(|err: PyErr| StarDistTrainError::Callback(err.to_string()))?;
        }
        Ok(())
    }
}

/// Train a StarDist2D model from paired image and ground-truth folders.
///
/// Args:
///     data_dir: Folder containing input images, or a dataset root when
///         `gt_dir` is omitted. Dataset roots may contain `train` and
///         `val`/`validation` folders.
///     gt_dir: Optional folder containing instance-label masks with matching
///         stems. If omitted, `data_dir` is auto-detected as a dataset root or
///         same-folder image/mask layout.
///     output_dir: Optional artifact directory. When provided, the trained model
///         and config files are saved there.
///     config: Optional dict overriding `TrainingConfig2D` defaults.
///     valid_fraction: Fraction of samples reserved for validation.
///     image_channels: `"auto"`, `"grayscale"`, or `"rgb"`.
///     label_color_mode: `"auto"`, `"grayscale"`, or `"color_ids"`.
///     gpu: If true, train with Burn WebGPU. Otherwise train on CPU.
///     on_train_begin: Callable receiving a dict with planned epoch/step counts.
///     on_step_end: Callable receiving a dict after every optimizer step.
///     on_validation_end: Callable receiving epoch metrics and validation
///         previews after each epoch.
#[pyfunction]
#[pyo3(name = "train_stardist_2d_folder")]
#[pyo3(signature = (
    data_dir,
    gt_dir=None,
    output_dir=None,
    config=None,
    valid_fraction=None,
    image_channels=None,
    label_color_mode=None,
    gpu=None,
    on_train_begin=None,
    on_step_end=None,
    on_validation_end=None
))]
pub fn train_stardist_2d_folder<'py>(
    py: Python<'py>,
    data_dir: String,
    gt_dir: Option<String>,
    output_dir: Option<String>,
    config: Option<Bound<'py, PyDict>>,
    valid_fraction: Option<f32>,
    image_channels: Option<String>,
    label_color_mode: Option<String>,
    gpu: Option<bool>,
    on_train_begin: Option<Py<PyAny>>,
    on_step_end: Option<Py<PyAny>>,
    on_validation_end: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let config_image_channels = config_string(config.as_ref(), "image_channels")?;
    let image_channels = parse_image_channels_option(
        image_channels
            .as_deref()
            .or(config_image_channels.as_deref()),
    )?;
    let config_label_color_mode = config_string(config.as_ref(), "label_color_mode")?;
    let label_color_mode = parse_label_color_mode_option(
        label_color_mode
            .as_deref()
            .or(config_label_color_mode.as_deref()),
    )?;

    let mut train_config = TrainingConfig2D::default();
    let n_channel_in_was_explicit = config
        .as_ref()
        .map(|dict| dict.contains("n_channel_in").unwrap_or(false))
        .unwrap_or(false);
    if let Some(config_dict) = config.as_ref() {
        apply_training_config_dict(&mut train_config, config_dict)?;
    }

    let options = FolderDatasetOptions2D {
        image_channels,
        label_color_mode,
    };
    let validation_fraction = validation_fraction_from_inputs(config.as_ref(), valid_fraction)?;
    let (train_samples, valid_samples) = if let Some(gt_dir) = gt_dir {
        let samples = load_training_samples_from_folders_with_options(data_dir, gt_dir, options)
            .map_err(stardist_train_error_to_pyerr)?;
        split_train_valid_2d(samples, validation_fraction, train_config.seed)
            .map_err(stardist_train_error_to_pyerr)?
    } else {
        let split = load_training_dataset_from_folder_with_options(
            data_dir,
            options,
            validation_fraction,
            train_config.seed,
        )
        .map_err(stardist_train_error_to_pyerr)?;
        (split.train_samples, split.valid_samples)
    };

    if !n_channel_in_was_explicit {
        if let Some(first) = train_samples.first().or_else(|| valid_samples.first()) {
            train_config.n_channel_in = first.image.dim().0;
        }
    }

    let output_dir = output_dir.map(PathBuf::from);
    let use_gpu = gpu.unwrap_or(false);
    let mut callbacks =
        PythonTrainingCallbacks2D::new(on_train_begin, on_step_end, on_validation_end);

    let summary = py
        .detach(|| {
            if use_gpu {
                run_training_backend::<WgpuTrainBackend>(
                    train_samples,
                    valid_samples,
                    train_config,
                    output_dir,
                    &mut callbacks,
                )
            } else {
                run_training_backend::<CpuTrainBackend>(
                    train_samples,
                    valid_samples,
                    train_config,
                    output_dir,
                    &mut callbacks,
                )
            }
        })
        .map_err(stardist_train_error_to_pyerr)?;

    training_summary_to_pydict(py, &summary)
}

fn run_training_backend<B>(
    train_samples: Vec<TrainingSample2D>,
    valid_samples: Vec<TrainingSample2D>,
    config: TrainingConfig2D,
    output_dir: Option<PathBuf>,
    callbacks: &mut PythonTrainingCallbacks2D,
) -> Result<PythonTrainingSummary2D, StarDistTrainError>
where
    B: burn::tensor::backend::AutodiffBackend<FloatElem = f32, IntElem = i32>,
    B::Device: Default,
{
    let train_count = train_samples.len();
    let valid_count = valid_samples.len();
    let valid_ref = if valid_samples.is_empty() {
        None
    } else {
        Some(valid_samples.as_slice())
    };
    let device = Default::default();
    let result: TrainingResult2D<B::InnerBackend> = train_stardist_2d_with_callbacks::<B, _>(
        &train_samples,
        valid_ref,
        config,
        device,
        callbacks,
    )?;

    if let Some(output_dir) = output_dir.as_ref() {
        save_stardist_2d(result.model, &result.config, output_dir)?;
    }

    Ok(PythonTrainingSummary2D {
        history: result.history,
        config: result.config,
        output_dir,
        train_samples: train_count,
        valid_samples: valid_count,
    })
}

fn apply_training_config_dict(
    config: &mut TrainingConfig2D,
    dict: &Bound<'_, PyDict>,
) -> PyResult<()> {
    if let Some(value) = config_usize(Some(dict), "n_channel_in")? {
        config.n_channel_in = value;
    }
    if let Some(value) = config_usize(Some(dict), "n_rays")? {
        config.n_rays = value;
    }
    if let Some(value) = config_pair_usize(Some(dict), "grid")? {
        config.grid = value;
    }
    if let Some(value) = config_pair_usize(Some(dict), "patch_size")? {
        config.patch_size = value;
    }
    if let Some(value) = config_usize(Some(dict), "batch_size")? {
        config.batch_size = value;
    }
    if let Some(value) = config_usize(Some(dict), "epochs")? {
        config.epochs = value;
    }
    if let Some(value) = config_usize(Some(dict), "steps_per_epoch")? {
        config.steps_per_epoch = value;
    }
    if let Some(value) = config_usize(Some(dict), "validation_steps")? {
        config.validation_steps = value;
    }
    if let Some(value) = config_f64(Some(dict), "learning_rate")? {
        config.learning_rate = value;
    }
    if let Some(value) = config_f32(Some(dict), "weight_decay")? {
        config.weight_decay = value;
    }
    if let Some(value) = config_f32(Some(dict), "foreground_probability")? {
        config.foreground_probability = value;
    }
    if let Some(value) = config_f32(Some(dict), "background_reg")? {
        config.background_reg = value;
    }
    if let Some(value) = config_f32(Some(dict), "loss_prob_weight")? {
        config.loss_prob_weight = value;
    }
    if let Some(value) = config_f32(Some(dict), "loss_dist_weight")? {
        config.loss_dist_weight = value;
    }
    if let Some(value) = config_f32(Some(dict), "prob_threshold")? {
        config.prob_threshold = value;
    }
    if let Some(value) = config_f32(Some(dict), "nms_threshold")? {
        config.nms_threshold = value;
    }
    if let Some(value) = config_u64(Some(dict), "seed")? {
        config.seed = value;
    }
    if let Some(value) = config_usize(Some(dict), "validation_preview_count")? {
        config.validation_preview_count = value;
    }
    if let Some(value) = config_bool(Some(dict), "shape_completion")? {
        config.shape_completion = value;
    }
    if let Some(value) = config_usize(Some(dict), "completion_crop")? {
        config.completion_crop = value;
    }

    apply_normalization_config(config, dict)?;
    apply_lr_schedule_config(config, dict)?;
    apply_augment_config(&mut config.augment, dict)?;
    Ok(())
}

fn apply_normalization_config(
    config: &mut TrainingConfig2D,
    dict: &Bound<'_, PyDict>,
) -> PyResult<()> {
    if config_bool(Some(dict), "normalize_percentile")?.unwrap_or(false) {
        config.normalization = Normalization2D::Percentile {
            pmin: 1.0,
            pmax: 99.8,
        };
    }
    if let Some(value) = config_string(Some(dict), "normalization")? {
        let value = value.to_ascii_lowercase();
        match value.as_str() {
            "none" => config.normalization = Normalization2D::None,
            "percentile" => {
                config.normalization = Normalization2D::Percentile {
                    pmin: 1.0,
                    pmax: 99.8,
                };
            }
            _ => {
                return Err(PyValueError::new_err(
                    "normalization must be 'none' or 'percentile'",
                ));
            }
        }
    }
    if let Some([pmin, pmax]) = config_pair_f32(Some(dict), "normalization_percentiles")? {
        config.normalization = Normalization2D::Percentile { pmin, pmax };
    }
    Ok(())
}

fn apply_lr_schedule_config(
    config: &mut TrainingConfig2D,
    dict: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let Some(schedule) = config_string(Some(dict), "lr_schedule")? else {
        return Ok(());
    };
    let min_learning_rate = config_f64(Some(dict), "min_learning_rate")?.unwrap_or(0.0);
    let schedule = schedule.to_ascii_lowercase();
    config.lr_schedule = match schedule.as_str() {
        "constant" => LearningRateSchedule2D::Constant,
        "reduce_on_plateau" | "plateau" => LearningRateSchedule2D::ReduceOnPlateau {
            factor: config_f64(Some(dict), "lr_factor")?.unwrap_or(0.5),
            patience: config_usize(Some(dict), "lr_patience")?.unwrap_or(40),
            min_delta: config_f32(Some(dict), "lr_min_delta")?.unwrap_or(0.0),
            min_learning_rate,
        },
        "cosine" | "cosine_annealing" => {
            LearningRateSchedule2D::CosineAnnealing { min_learning_rate }
        }
        "polynomial" | "polynomial_decay" | "poly" => LearningRateSchedule2D::PolynomialDecay {
            power: config_f64(Some(dict), "poly_power")?.unwrap_or(0.9),
            min_learning_rate,
        },
        _ => {
            return Err(PyValueError::new_err(
                "lr_schedule must be 'constant', 'reduce_on_plateau', 'cosine', or 'polynomial'",
            ));
        }
    };
    Ok(())
}

fn apply_augment_config(augment: &mut AugmentConfig2D, dict: &Bound<'_, PyDict>) -> PyResult<()> {
    if let Some(value) = config_bool(Some(dict), "flip_y")? {
        augment.flip_y = value;
    }
    if let Some(value) = config_bool(Some(dict), "flip_x")? {
        augment.flip_x = value;
    }
    if let Some(value) = config_bool(Some(dict), "rotate90")? {
        augment.rotate90 = value;
    }
    if let Some(value) = config_f32(Some(dict), "intensity_scale")? {
        augment.intensity_scale = value;
    }
    if let Some(value) = config_f32(Some(dict), "intensity_shift")? {
        augment.intensity_shift = value;
    }
    if let Some(value) = config_f32(Some(dict), "gaussian_noise_std")? {
        augment.gaussian_noise_std = value;
    }
    Ok(())
}

fn validation_fraction_from_inputs(
    dict: Option<&Bound<'_, PyDict>>,
    explicit: Option<f32>,
) -> PyResult<f32> {
    let mut value = explicit;
    for key in [
        "valid_fraction",
        "val_fraction",
        "validation_fraction",
        "val_percentage",
        "validation_percentage",
    ] {
        if value.is_none() {
            value = config_f32(dict, key)?;
        }
    }

    normalize_validation_fraction(value.unwrap_or(DEFAULT_VALID_FRACTION_2D))
}

fn normalize_validation_fraction(value: f32) -> PyResult<f32> {
    let fraction = if value > 1.0 && value <= 100.0 {
        value / 100.0
    } else {
        value
    };

    if (0.0..1.0).contains(&fraction) {
        Ok(fraction)
    } else {
        Err(PyValueError::new_err(
            "valid_fraction must be in [0, 1), or val_percentage must be in [0, 100)",
        ))
    }
}

fn parse_image_channels_option(value: Option<&str>) -> PyResult<ImageChannels2D> {
    let value = value.unwrap_or("auto").to_ascii_lowercase();
    match value.as_str() {
        "auto" => Ok(ImageChannels2D::Auto),
        "gray" | "grey" | "grayscale" => Ok(ImageChannels2D::Grayscale),
        "rgb" => Ok(ImageChannels2D::Rgb),
        _ => Err(PyValueError::new_err(
            "image_channels must be 'auto', 'grayscale', or 'rgb'",
        )),
    }
}

fn parse_label_color_mode_option(value: Option<&str>) -> PyResult<LabelColorMode2D> {
    let value = value.unwrap_or("auto").to_ascii_lowercase();
    match value.as_str() {
        "auto" => Ok(LabelColorMode2D::Auto),
        "gray" | "grey" | "grayscale" => Ok(LabelColorMode2D::Grayscale),
        "color_ids" | "color" | "rgb" => Ok(LabelColorMode2D::ColorIds),
        _ => Err(PyValueError::new_err(
            "label_color_mode must be 'auto', 'grayscale', or 'color_ids'",
        )),
    }
}

fn config_string(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<String>> {
    get_config_value(dict, key)
}

fn config_bool(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<bool>> {
    get_config_value(dict, key)
}

fn config_usize(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<usize>> {
    get_config_value(dict, key)
}

fn config_u64(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<u64>> {
    get_config_value(dict, key)
}

fn config_f32(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<f32>> {
    get_config_value(dict, key)
}

fn config_f64(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<f64>> {
    get_config_value(dict, key)
}

fn get_config_value<'py, T>(dict: Option<&Bound<'py, PyDict>>, key: &str) -> PyResult<Option<T>>
where
    T: FromPyObjectOwned<'py>,
{
    let Some(dict) = dict else {
        return Ok(None);
    };
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    value.extract::<T>().map(Some).map_err(Into::into)
}

fn config_pair_usize(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<[usize; 2]>> {
    let Some(values) = config_vec_usize(dict, key)? else {
        return Ok(None);
    };
    if values.len() != 2 {
        return Err(PyValueError::new_err(format!("{key} must have length 2")));
    }
    Ok(Some([values[0], values[1]]))
}

fn config_pair_f32(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<[f32; 2]>> {
    let Some(values) = config_vec_f32(dict, key)? else {
        return Ok(None);
    };
    if values.len() != 2 {
        return Err(PyValueError::new_err(format!("{key} must have length 2")));
    }
    Ok(Some([values[0], values[1]]))
}

fn config_vec_usize(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<Vec<usize>>> {
    get_config_value(dict, key)
}

fn config_vec_f32(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<Vec<f32>>> {
    get_config_value(dict, key)
}

fn training_plan_to_pydict<'py>(
    py: Python<'py>,
    plan: &TrainingPlan2D,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("epochs", plan.epochs)?;
    dict.set_item("steps_per_epoch", plan.steps_per_epoch)?;
    dict.set_item("validation_steps", plan.validation_steps)?;
    dict.set_item("total_steps", plan.total_steps)?;
    dict.set_item("train_samples", plan.train_samples)?;
    dict.set_item("valid_samples", plan.valid_samples)?;
    dict.set_item("batch_size", plan.batch_size)?;
    dict.set_item("patch_size", plan.patch_size)?;
    dict.set_item("grid", plan.grid)?;
    dict.set_item("n_rays", plan.n_rays)?;
    dict.set_item("validation_preview_count", plan.validation_preview_count)?;
    Ok(dict)
}

fn step_metrics_to_pydict<'py>(
    py: Python<'py>,
    metrics: &StepMetrics2D,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("epoch", metrics.epoch)?;
    dict.set_item("step", metrics.step)?;
    dict.set_item("global_step", metrics.global_step)?;
    dict.set_item("learning_rate", metrics.learning_rate)?;
    dict.set_item("loss_total", metrics.total)?;
    dict.set_item("loss_prob", metrics.prob)?;
    dict.set_item("loss_dist", metrics.dist)?;
    Ok(dict)
}

fn epoch_event_to_pydict<'py>(
    py: Python<'py>,
    event: &EpochEndEvent2D,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = epoch_metrics_to_pydict(py, &event.metrics)?;
    dict.set_item("validation_ran", event.validation_ran)?;
    dict.set_item(
        "previews",
        validation_previews_to_pylist(py, &event.previews)?,
    )?;
    Ok(dict)
}

fn epoch_metrics_to_pydict<'py>(
    py: Python<'py>,
    metrics: &EpochMetrics,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("epoch", metrics.epoch)?;
    dict.set_item("learning_rate", metrics.learning_rate)?;
    dict.set_item("train_total", metrics.train_total)?;
    dict.set_item("train_prob", metrics.train_prob)?;
    dict.set_item("train_dist", metrics.train_dist)?;
    set_optional_f32(py, &dict, "valid_total", metrics.valid_total)?;
    set_optional_f32(py, &dict, "valid_prob", metrics.valid_prob)?;
    set_optional_f32(py, &dict, "valid_dist", metrics.valid_dist)?;
    Ok(dict)
}

fn validation_previews_to_pylist<'py>(
    py: Python<'py>,
    previews: &[ValidationPreview2D],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for preview in previews {
        let dict = PyDict::new(py);
        dict.set_item("sample_index", preview.sample_index)?;
        dict.set_item("image", preview.image.clone().into_pyarray(py))?;
        dict.set_item("labels", preview.labels.clone().into_pyarray(py))?;
        dict.set_item("prediction", preview.prediction.clone().into_pyarray(py))?;
        dict.set_item("prob", preview.prob.clone().into_pyarray(py))?;
        list.append(dict)?;
    }
    Ok(list)
}

fn training_summary_to_pydict<'py>(
    py: Python<'py>,
    summary: &PythonTrainingSummary2D,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let history = PyList::empty(py);
    for metrics in &summary.history {
        history.append(epoch_metrics_to_pydict(py, metrics)?)?;
    }
    dict.set_item("history", history)?;
    dict.set_item("config", training_config_to_pydict(py, &summary.config)?)?;
    dict.set_item("train_samples", summary.train_samples)?;
    dict.set_item("valid_samples", summary.valid_samples)?;
    match &summary.output_dir {
        Some(path) => dict.set_item("output_dir", path.display().to_string())?,
        None => dict.set_item("output_dir", py.None())?,
    }
    Ok(dict)
}

fn training_config_to_pydict<'py>(
    py: Python<'py>,
    config: &TrainingConfig2D,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("n_channel_in", config.n_channel_in)?;
    dict.set_item("n_rays", config.n_rays)?;
    dict.set_item("grid", config.grid)?;
    dict.set_item("patch_size", config.patch_size)?;
    dict.set_item("batch_size", config.batch_size)?;
    dict.set_item("epochs", config.epochs)?;
    dict.set_item("steps_per_epoch", config.steps_per_epoch)?;
    dict.set_item("validation_steps", config.validation_steps)?;
    dict.set_item("learning_rate", config.learning_rate)?;
    dict.set_item("validation_preview_count", config.validation_preview_count)?;
    dict.set_item("prob_threshold", config.prob_threshold)?;
    dict.set_item("nms_threshold", config.nms_threshold)?;
    Ok(dict)
}

fn set_optional_f32<'py>(
    py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    key: &str,
    value: Option<f32>,
) -> PyResult<()> {
    match value {
        Some(value) => dict.set_item(key, value),
        None => dict.set_item(key, py.None()),
    }
}
