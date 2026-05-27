use std::fs;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend;
use cellcast::networks::stardist::trainable_2d::{TrainableStarDist2D, TrainableStarDist2DConfig};
use cellcast::training::io_2d::{
    load_training_dataset_from_folder_with_options,
    load_training_samples_from_folders_with_options, split_train_valid_2d, FolderDatasetOptions2D,
    ImageChannels2D, LabelColorMode2D,
};
use cellcast::training::stardist_2d::{
    predict_stardist_2d_batch_with_thresholds, predict_stardist_2d_with_thresholds,
    save_stardist_2d, train_stardist_2d_with_callbacks, AugmentConfig2D, CpuInferBackend,
    CpuTrainBackend, EpochEndEvent2D, EpochMetrics, LearningRateSchedule2D, Normalization2D,
    PythonStarDist2DConfig, PythonThresholds, StarDistTrainError, StepMetrics2D,
    TrainingCallbacks2D, TrainingConfig2D, TrainingPlan2D, TrainingResult2D, TrainingSample2D,
    ValidationPreview2D, WgpuInferBackend, WgpuTrainBackend,
};
use numpy::ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4};
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4};
use pyo3::exceptions::PyTypeError;
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

enum StarDist2DLoadSource {
    New,
    ConfigJson(PathBuf),
    Trained(TrainedStarDist2DSource),
}

enum TrainedStarDist2DSource {
    ArtifactDir(PathBuf),
    ModelFile(PathBuf),
}

enum PyImageInput2D {
    Single(Array3<f32>),
    BatchBcyx(Array4<f32>),
}

enum PyPredictionOutput2D {
    Single(Array2<u64>),
    Batch(Array3<u64>),
}

enum LoadedStarDist2DModel {
    Cpu {
        model: TrainableStarDist2D<CpuInferBackend>,
        config: TrainingConfig2D,
        device: <CpuInferBackend as Backend>::Device,
    },
    Wgpu {
        model: TrainableStarDist2D<WgpuInferBackend>,
        config: TrainingConfig2D,
        device: <WgpuInferBackend as Backend>::Device,
    },
}

/// Loaded StarDist2D model for repeated Python inference.
///
/// Delete the Python object to release the Rust model and backend resources:
/// `del model`.
#[pyclass(name = "StarDist2DModel", unsendable)]
pub struct PyStarDist2DModel {
    inner: LoadedStarDist2DModel,
}

#[pymethods]
impl PyStarDist2DModel {
    #[pyo3(signature = (data, prob_threshold=None, nms_threshold=None, axis=None))]
    pub fn predict<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyAny>,
        prob_threshold: Option<f32>,
        nms_threshold: Option<f32>,
        axis: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let input = py_image_input_2d(&data, axis)?;
        let output = self
            .inner
            .predict_input(&input, prob_threshold, nms_threshold)
            .map_err(stardist_train_error_to_pyerr)?;
        Ok(prediction_output_to_py(py, output))
    }

    #[getter]
    pub fn gpu(&self) -> bool {
        matches!(self.inner, LoadedStarDist2DModel::Wgpu { .. })
    }

    #[getter]
    pub fn n_channel_in(&self) -> usize {
        self.inner.config().n_channel_in
    }

    #[getter]
    pub fn n_rays(&self) -> usize {
        self.inner.config().n_rays
    }

    #[getter]
    pub fn grid(&self) -> [usize; 2] {
        self.inner.config().grid
    }

    #[getter]
    pub fn prob_threshold(&self) -> f32 {
        self.inner.config().prob_threshold
    }

    #[setter]
    pub fn set_prob_threshold(&mut self, value: f32) -> PyResult<()> {
        validate_threshold("prob_threshold", value)?;
        self.inner.config_mut().prob_threshold = value;
        Ok(())
    }

    #[getter]
    pub fn nms_threshold(&self) -> f32 {
        self.inner.config().nms_threshold
    }

    #[setter]
    pub fn set_nms_threshold(&mut self, value: f32) -> PyResult<()> {
        validate_threshold("nms_threshold", value)?;
        self.inner.config_mut().nms_threshold = value;
        Ok(())
    }

    #[pyo3(signature = (prob_threshold=None, nms_threshold=None))]
    pub fn set_thresholds(
        &mut self,
        prob_threshold: Option<f32>,
        nms_threshold: Option<f32>,
    ) -> PyResult<()> {
        if let Some(value) = prob_threshold {
            validate_threshold("prob_threshold", value)?;
            self.inner.config_mut().prob_threshold = value;
        }
        if let Some(value) = nms_threshold {
            validate_threshold("nms_threshold", value)?;
            self.inner.config_mut().nms_threshold = value;
        }
        Ok(())
    }

    pub fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        training_config_to_pydict(py, self.inner.config())
    }
}

impl LoadedStarDist2DModel {
    fn config(&self) -> &TrainingConfig2D {
        match self {
            Self::Cpu { config, .. } | Self::Wgpu { config, .. } => config,
        }
    }

    fn config_mut(&mut self) -> &mut TrainingConfig2D {
        match self {
            Self::Cpu { config, .. } | Self::Wgpu { config, .. } => config,
        }
    }

    fn predict_input(
        &self,
        input: &PyImageInput2D,
        prob_threshold: Option<f32>,
        nms_threshold: Option<f32>,
    ) -> Result<PyPredictionOutput2D, StarDistTrainError> {
        match self {
            Self::Cpu {
                model,
                config,
                device,
            } => {
                let prob_threshold = prob_threshold.unwrap_or(config.prob_threshold);
                let nms_threshold = nms_threshold.unwrap_or(config.nms_threshold);
                predict_with_model_input(
                    model,
                    config,
                    device,
                    input,
                    prob_threshold,
                    nms_threshold,
                )
            }
            Self::Wgpu {
                model,
                config,
                device,
            } => {
                let prob_threshold = prob_threshold.unwrap_or(config.prob_threshold);
                let nms_threshold = nms_threshold.unwrap_or(config.nms_threshold);
                predict_with_model_input(
                    model,
                    config,
                    device,
                    input,
                    prob_threshold,
                    nms_threshold,
                )
            }
        }
    }
}

/// Create a randomly initialized StarDist2D model.
///
/// This does not load trained weights. It is useful for inspecting config,
/// checking device availability, or future workflows that train/update an
/// already-created model.
#[pyfunction]
#[pyo3(name = "new_stardist_2d")]
#[pyo3(signature = (config=None, gpu=None))]
pub fn new_stardist_2d(
    config: Option<Bound<'_, PyDict>>,
    gpu: Option<Bound<'_, PyAny>>,
) -> PyResult<PyStarDist2DModel> {
    let config = training_config_from_pydict(TrainingConfig2D::default(), config.as_ref())?;
    new_stardist_2d_from_config(config, parse_bool_argument(gpu.as_ref(), "gpu")?)
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

/// Predict instance labels with a StarDist2D model saved by
/// `train_stardist_2d_folder`.
///
/// Args:
///     model_dir: Artifact directory produced by training.
///     data: Input image. A 2D array is treated as one grayscale channel. A 3D
///         array is interpreted according to `axis`.
///     prob_threshold: Optional object probability threshold. If omitted, the
///         saved training config threshold is used.
///     nms_threshold: Optional NMS overlap threshold. If omitted, the saved
///         training config threshold is used.
///     axis: Channel axis for 3D input arrays. Defaults to the last axis.
///     gpu: If true, use the Burn WebGPU backend. Otherwise use CPU.
#[pyfunction]
#[pyo3(name = "predict_stardist_2d")]
#[pyo3(signature = (
    model_dir,
    data,
    prob_threshold=None,
    nms_threshold=None,
    axis=None,
    gpu=None
))]
pub fn predict_stardist_2d_saved<'py>(
    py: Python<'py>,
    model_dir: String,
    data: Bound<'py, PyAny>,
    prob_threshold: Option<f32>,
    nms_threshold: Option<f32>,
    axis: Option<usize>,
    gpu: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    let input = py_image_input_2d(&data, axis)?;
    let use_gpu = gpu.unwrap_or(false);
    let output = py
        .detach(|| {
            if use_gpu {
                predict_saved_backend::<WgpuInferBackend>(
                    &model_dir,
                    &input,
                    prob_threshold,
                    nms_threshold,
                )
            } else {
                predict_saved_backend::<CpuInferBackend>(
                    &model_dir,
                    &input,
                    prob_threshold,
                    nms_threshold,
                )
            }
        })
        .map_err(stardist_train_error_to_pyerr)?;
    Ok(prediction_output_to_py(py, output))
}

/// Load or create a StarDist2D model once for repeated inference.
///
/// If `source` is omitted, a new randomly initialized model is created from the
/// default config plus optional config overrides. If `source` is a JSON config,
/// a new randomly initialized model is created from that config. If `source` is
/// an artifact directory or `model.mpk`, trained weights are loaded and only
/// safe inference overrides, such as thresholds, are applied.
#[pyfunction]
#[pyo3(name = "load_stardist_2d")]
#[pyo3(signature = (source=None, gpu=None, config=None, model_dir=None))]
pub fn load_stardist_2d_saved(
    source: Option<String>,
    gpu: Option<Bound<'_, PyAny>>,
    config: Option<Bound<'_, PyDict>>,
    model_dir: Option<String>,
) -> PyResult<PyStarDist2DModel> {
    let source = merge_model_source(source, model_dir)?;
    let source =
        classify_stardist_2d_source(source.as_deref()).map_err(stardist_train_error_to_pyerr)?;
    let use_gpu = parse_bool_argument(gpu.as_ref(), "gpu")?;

    match source {
        StarDist2DLoadSource::New => {
            let config = training_config_from_pydict(TrainingConfig2D::default(), config.as_ref())?;
            new_stardist_2d_from_config(config, use_gpu)
        }
        StarDist2DLoadSource::ConfigJson(path) => {
            let base =
                training_config_from_json_path(&path).map_err(stardist_train_error_to_pyerr)?;
            let config = training_config_from_pydict(base, config.as_ref())?;
            new_stardist_2d_from_config(config, use_gpu)
        }
        StarDist2DLoadSource::Trained(source) => {
            if use_gpu {
                let device = Default::default();
                let (model, config) = load_trained_stardist_2d_backend::<WgpuInferBackend>(
                    &source,
                    config.as_ref(),
                    &device,
                )?;
                Ok(PyStarDist2DModel {
                    inner: LoadedStarDist2DModel::Wgpu {
                        model,
                        config,
                        device,
                    },
                })
            } else {
                let device = Default::default();
                let (model, config) = load_trained_stardist_2d_backend::<CpuInferBackend>(
                    &source,
                    config.as_ref(),
                    &device,
                )?;
                Ok(PyStarDist2DModel {
                    inner: LoadedStarDist2DModel::Cpu {
                        model,
                        config,
                        device,
                    },
                })
            }
        }
    }
}

/// Alias for clarity when calling from Python.
#[pyfunction]
#[pyo3(name = "predict_trained_stardist_2d")]
#[pyo3(signature = (
    model_dir,
    data,
    prob_threshold=None,
    nms_threshold=None,
    axis=None,
    gpu=None
))]
pub fn predict_trained_stardist_2d<'py>(
    py: Python<'py>,
    model_dir: String,
    data: Bound<'py, PyAny>,
    prob_threshold: Option<f32>,
    nms_threshold: Option<f32>,
    axis: Option<usize>,
    gpu: Option<bool>,
) -> PyResult<Bound<'py, PyAny>> {
    predict_stardist_2d_saved(
        py,
        model_dir,
        data,
        prob_threshold,
        nms_threshold,
        axis,
        gpu,
    )
}

fn new_stardist_2d_from_config(
    config: TrainingConfig2D,
    use_gpu: bool,
) -> PyResult<PyStarDist2DModel> {
    if use_gpu {
        let device = Default::default();
        WgpuInferBackend::seed(&device, config.seed);
        let model_config =
            TrainableStarDist2DConfig::new(config.n_channel_in, config.n_rays, config.grid);
        let model = model_config.init::<WgpuInferBackend>(&device);
        Ok(PyStarDist2DModel {
            inner: LoadedStarDist2DModel::Wgpu {
                model,
                config,
                device,
            },
        })
    } else {
        let device = Default::default();
        CpuInferBackend::seed(&device, config.seed);
        let model_config =
            TrainableStarDist2DConfig::new(config.n_channel_in, config.n_rays, config.grid);
        let model = model_config.init::<CpuInferBackend>(&device);
        Ok(PyStarDist2DModel {
            inner: LoadedStarDist2DModel::Cpu {
                model,
                config,
                device,
            },
        })
    }
}

fn load_trained_stardist_2d_backend<B>(
    source: &TrainedStarDist2DSource,
    config_overrides: Option<&Bound<'_, PyDict>>,
    device: &B::Device,
) -> PyResult<(TrainableStarDist2D<B>, TrainingConfig2D)>
where
    B: Backend<FloatElem = f32, IntElem = i32>,
{
    let (model, mut config) = load_trained_stardist_2d_backend_result::<B>(source, device)
        .map_err(stardist_train_error_to_pyerr)?;
    apply_trained_model_config_overrides(&mut config, config_overrides)?;
    config.validate().map_err(stardist_train_error_to_pyerr)?;
    Ok((model, config))
}

fn load_trained_stardist_2d_backend_result<B>(
    source: &TrainedStarDist2DSource,
    device: &B::Device,
) -> Result<(TrainableStarDist2D<B>, TrainingConfig2D), StarDistTrainError>
where
    B: Backend<FloatElem = f32, IntElem = i32>,
{
    let (artifact_dir, model_stem) = match source {
        TrainedStarDist2DSource::ArtifactDir(artifact_dir) => {
            ensure_model_file_exists(artifact_dir)?;
            (artifact_dir.as_path(), artifact_dir.join("model"))
        }
        TrainedStarDist2DSource::ModelFile(model_file) => {
            let artifact_dir = model_file.parent().unwrap_or_else(|| Path::new("."));
            (artifact_dir, compact_recorder_stem(model_file)?)
        }
    };

    let config = read_saved_training_config(artifact_dir)?;
    let model_config =
        TrainableStarDist2DConfig::new(config.n_channel_in, config.n_rays, config.grid);
    let model = model_config.init::<B>(device);
    let recorder = CompactRecorder::new();
    let model = model.load_file(model_stem, &recorder, device)?;
    Ok((model, config))
}

fn merge_model_source(
    source: Option<String>,
    model_dir: Option<String>,
) -> PyResult<Option<String>> {
    match (source, model_dir) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "pass only one of source or model_dir to load_stardist_2d",
        )),
        (Some(source), None) => Ok(Some(source)),
        (None, Some(model_dir)) => Ok(Some(model_dir)),
        (None, None) => Ok(None),
    }
}

fn classify_stardist_2d_source(
    source: Option<&str>,
) -> Result<StarDist2DLoadSource, StarDistTrainError> {
    let Some(source) = source else {
        return Ok(StarDist2DLoadSource::New);
    };
    let path = PathBuf::from(source);
    if path.is_dir() {
        return Ok(StarDist2DLoadSource::Trained(
            TrainedStarDist2DSource::ArtifactDir(path),
        ));
    }
    if is_json_path(&path) {
        if path.is_file() {
            return Ok(StarDist2DLoadSource::ConfigJson(path));
        }
        return Err(StarDistTrainError::InvalidConfig(format!(
            "StarDist2D config JSON does not exist: {}",
            path.display()
        )));
    }
    classify_trained_stardist_2d_source(&path).map(StarDist2DLoadSource::Trained)
}

fn classify_trained_stardist_2d_source(
    path: &Path,
) -> Result<TrainedStarDist2DSource, StarDistTrainError> {
    if path.is_dir() {
        return Ok(TrainedStarDist2DSource::ArtifactDir(path.to_path_buf()));
    }
    if is_mpk_path(path) {
        if path.is_file() {
            return Ok(TrainedStarDist2DSource::ModelFile(path.to_path_buf()));
        }
        return Err(StarDistTrainError::InvalidConfig(format!(
            "StarDist2D model file does not exist: {}",
            path.display()
        )));
    }
    Err(StarDistTrainError::InvalidConfig(format!(
        "load_stardist_2d source must be None, a config .json file, an artifact directory, or a model .mpk file; got {}",
        path.display()
    )))
}

fn read_saved_training_config(artifact_dir: &Path) -> Result<TrainingConfig2D, StarDistTrainError> {
    let cellcast_config_path = artifact_dir.join("cellcast_config.json");
    let python_config_path = artifact_dir.join("config.json");
    let mut config = if cellcast_config_path.is_file() {
        training_config_from_json_path_result(&cellcast_config_path)?
    } else if python_config_path.is_file() {
        training_config_from_json_path_result(&python_config_path)?
    } else {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "cannot load trained StarDist2D weights because no cellcast_config.json or config.json was found next to the model in {}",
            artifact_dir.display()
        )));
    };

    apply_saved_thresholds(artifact_dir, &mut config)?;
    config.validate()?;
    Ok(config)
}

fn apply_saved_thresholds(
    artifact_dir: &Path,
    config: &mut TrainingConfig2D,
) -> Result<(), StarDistTrainError> {
    let thresholds_path = artifact_dir.join("thresholds.json");
    if thresholds_path.is_file() {
        let thresholds: PythonThresholds = serde_json::from_slice(&fs::read(thresholds_path)?)?;
        config.prob_threshold = thresholds.prob;
        config.nms_threshold = thresholds.nms;
    }
    Ok(())
}

fn ensure_model_file_exists(artifact_dir: &Path) -> Result<(), StarDistTrainError> {
    let model_path = artifact_dir.join("model.mpk");
    if !model_path.is_file() {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "StarDist2D artifact directory {} does not contain model.mpk",
            artifact_dir.display()
        )));
    }
    Ok(())
}

fn compact_recorder_stem(model_file: &Path) -> Result<PathBuf, StarDistTrainError> {
    if !is_mpk_path(model_file) {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "StarDist2D model file must end in .mpk: {}",
            model_file.display()
        )));
    }
    let Some(stem) = model_file.file_stem() else {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "invalid StarDist2D model file path: {}",
            model_file.display()
        )));
    };
    Ok(model_file.with_file_name(stem))
}

fn is_json_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

fn is_mpk_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.eq_ignore_ascii_case("mpk"))
        .unwrap_or(false)
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

fn predict_saved_backend<B>(
    model_source: &str,
    input: &PyImageInput2D,
    prob_threshold: Option<f32>,
    nms_threshold: Option<f32>,
) -> Result<PyPredictionOutput2D, StarDistTrainError>
where
    B: Backend<FloatElem = f32, IntElem = i32>,
    B::Device: Default,
{
    let device = Default::default();
    let source = classify_trained_stardist_2d_source(Path::new(model_source))?;
    let (model, config) = load_trained_stardist_2d_backend_result::<B>(&source, &device)?;
    let prob_threshold = prob_threshold.unwrap_or(config.prob_threshold);
    let nms_threshold = nms_threshold.unwrap_or(config.nms_threshold);
    predict_with_model_input(
        &model,
        &config,
        &device,
        input,
        prob_threshold,
        nms_threshold,
    )
}

fn predict_with_model_input<B>(
    model: &TrainableStarDist2D<B>,
    config: &TrainingConfig2D,
    device: &B::Device,
    input: &PyImageInput2D,
    prob_threshold: f32,
    nms_threshold: f32,
) -> Result<PyPredictionOutput2D, StarDistTrainError>
where
    B: Backend<FloatElem = f32, IntElem = i32>,
{
    match input {
        PyImageInput2D::Single(image) => {
            let labels = predict_stardist_2d_with_thresholds(
                model,
                image,
                config,
                prob_threshold,
                nms_threshold,
                device,
            )?;
            Ok(PyPredictionOutput2D::Single(labels))
        }
        PyImageInput2D::BatchBcyx(batch) => {
            let labels = predict_stardist_2d_batch_with_thresholds(
                model,
                batch,
                config,
                prob_threshold,
                nms_threshold,
                device,
            )?;
            Ok(PyPredictionOutput2D::Batch(labels))
        }
    }
}

fn prediction_output_to_py<'py>(
    py: Python<'py>,
    output: PyPredictionOutput2D,
) -> Bound<'py, PyAny> {
    match output {
        PyPredictionOutput2D::Single(labels) => labels.into_pyarray(py).into_any(),
        PyPredictionOutput2D::Batch(labels) => labels.into_pyarray(py).into_any(),
    }
}

fn py_image_input_2d(data: &Bound<'_, PyAny>, axis: Option<usize>) -> PyResult<PyImageInput2D> {
    if let Ok(arr) = data.extract::<PyReadonlyArray4<u8>>() {
        array4_bcyx_to_batch(arr.as_array(), axis).map(PyImageInput2D::BatchBcyx)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray4<u16>>() {
        array4_bcyx_to_batch(arr.as_array(), axis).map(PyImageInput2D::BatchBcyx)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray4<u64>>() {
        array4_bcyx_to_batch(arr.as_array(), axis).map(PyImageInput2D::BatchBcyx)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray4<f32>>() {
        array4_bcyx_to_batch(arr.as_array(), axis).map(PyImageInput2D::BatchBcyx)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray4<f64>>() {
        array4_bcyx_to_batch(arr.as_array(), axis).map(PyImageInput2D::BatchBcyx)
    } else {
        py_image_to_channel_first(data, axis).map(PyImageInput2D::Single)
    }
}

fn py_image_to_channel_first(
    data: &Bound<'_, PyAny>,
    axis: Option<usize>,
) -> PyResult<Array3<f32>> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<u8>>() {
        Ok(array2_to_channel_first(arr.as_array()))
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u16>>() {
        Ok(array2_to_channel_first(arr.as_array()))
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<u64>>() {
        Ok(array2_to_channel_first(arr.as_array()))
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f32>>() {
        Ok(array2_to_channel_first(arr.as_array()))
    } else if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        Ok(array2_to_channel_first(arr.as_array()))
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u8>>() {
        array3_to_channel_first(arr.as_array(), axis)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u16>>() {
        array3_to_channel_first(arr.as_array(), axis)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<u64>>() {
        array3_to_channel_first(arr.as_array(), axis)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f32>>() {
        array3_to_channel_first(arr.as_array(), axis)
    } else if let Ok(arr) = data.extract::<PyReadonlyArray3<f64>>() {
        array3_to_channel_first(arr.as_array(), axis)
    } else {
        Err(PyTypeError::new_err(
            "data must be a 2D [Y, X], 3D single-image, or 4D [B, C, Y, X] NumPy array with dtype u8, u16, u64, f32, or f64",
        ))
    }
}

trait ImageScalar2D {
    fn to_f32(self) -> f32;
}

impl ImageScalar2D for u8 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl ImageScalar2D for u16 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl ImageScalar2D for u64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

impl ImageScalar2D for f32 {
    fn to_f32(self) -> f32 {
        self
    }
}

impl ImageScalar2D for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
}

fn array2_to_channel_first<T>(view: ArrayView2<'_, T>) -> Array3<f32>
where
    T: Copy + ImageScalar2D,
{
    let (height, width) = view.dim();
    let mut image = Array3::<f32>::zeros((1, height, width));
    for y in 0..height {
        for x in 0..width {
            image[[0, y, x]] = view[[y, x]].to_f32();
        }
    }
    image
}

fn array3_to_channel_first<T>(view: ArrayView3<'_, T>, axis: Option<usize>) -> PyResult<Array3<f32>>
where
    T: Copy + ImageScalar2D,
{
    let axis = axis.unwrap_or(2);
    let (d0, d1, d2) = view.dim();
    match axis {
        0 => {
            let mut image = Array3::<f32>::zeros((d0, d1, d2));
            for c in 0..d0 {
                for y in 0..d1 {
                    for x in 0..d2 {
                        image[[c, y, x]] = view[[c, y, x]].to_f32();
                    }
                }
            }
            Ok(image)
        }
        1 => {
            let mut image = Array3::<f32>::zeros((d1, d0, d2));
            for y in 0..d0 {
                for c in 0..d1 {
                    for x in 0..d2 {
                        image[[c, y, x]] = view[[y, c, x]].to_f32();
                    }
                }
            }
            Ok(image)
        }
        2 => {
            let mut image = Array3::<f32>::zeros((d2, d0, d1));
            for y in 0..d0 {
                for x in 0..d1 {
                    for c in 0..d2 {
                        image[[c, y, x]] = view[[y, x, c]].to_f32();
                    }
                }
            }
            Ok(image)
        }
        _ => Err(PyValueError::new_err("axis must be 0, 1, or 2")),
    }
}

fn array4_bcyx_to_batch<T>(view: ArrayView4<'_, T>, axis: Option<usize>) -> PyResult<Array4<f32>>
where
    T: Copy + ImageScalar2D,
{
    if let Some(axis) = axis {
        if axis != 1 {
            return Err(PyValueError::new_err(
                "4D batch input must be [B, C, Y, X]; axis must be omitted or set to 1",
            ));
        }
    }
    let (batch, channels, height, width) = view.dim();
    let mut images = Array4::<f32>::zeros((batch, channels, height, width));
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    images[[b, c, y, x]] = view[[b, c, y, x]].to_f32();
                }
            }
        }
    }
    Ok(images)
}

fn training_config_from_pydict(
    mut config: TrainingConfig2D,
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<TrainingConfig2D> {
    if let Some(dict) = dict {
        apply_training_config_dict(&mut config, dict)?;
    }
    config.validate().map_err(stardist_train_error_to_pyerr)?;
    Ok(config)
}

fn training_config_from_json_path(path: &Path) -> Result<TrainingConfig2D, StarDistTrainError> {
    let mut config = training_config_from_json_path_result(path)?;
    apply_saved_thresholds(path.parent().unwrap_or_else(|| Path::new(".")), &mut config)?;
    config.validate()?;
    Ok(config)
}

fn training_config_from_json_path_result(
    path: &Path,
) -> Result<TrainingConfig2D, StarDistTrainError> {
    let bytes = fs::read(path)?;
    if let Ok(config) = serde_json::from_slice::<TrainingConfig2D>(&bytes) {
        return Ok(config);
    }
    if let Ok(python_config) = serde_json::from_slice::<PythonStarDist2DConfig>(&bytes) {
        return Ok(training_config_from_python_config(python_config));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    training_config_from_json_value(&value)
}

fn training_config_from_python_config(python_config: PythonStarDist2DConfig) -> TrainingConfig2D {
    let mut config = TrainingConfig2D::default();
    config.n_channel_in = python_config.n_channel_in;
    config.n_rays = python_config.n_rays;
    config.grid = python_config.grid;
    config.patch_size = python_config.train_patch_size;
    config.batch_size = python_config.train_batch_size;
    config.epochs = python_config.train_epochs;
    config.steps_per_epoch = python_config.train_steps_per_epoch;
    config.learning_rate = python_config.train_learning_rate;
    config.background_reg = python_config.train_background_reg;
    config.foreground_probability = python_config.train_foreground_only;
    config.loss_prob_weight = python_config.train_loss_weights[0];
    config.loss_dist_weight = python_config.train_loss_weights[1];
    config.shape_completion = python_config.train_shape_completion;
    config.completion_crop = python_config.train_completion_crop;
    config.lr_schedule = LearningRateSchedule2D::ReduceOnPlateau {
        factor: python_config.train_reduce_lr.factor as f64,
        patience: python_config.train_reduce_lr.patience,
        min_delta: python_config.train_reduce_lr.min_delta,
        min_learning_rate: 0.0,
    };
    config
}

fn training_config_from_json_value(
    value: &serde_json::Value,
) -> Result<TrainingConfig2D, StarDistTrainError> {
    let mut config = TrainingConfig2D::default();
    if let Some(value) = json_usize(value, "n_channel_in")? {
        config.n_channel_in = value;
    }
    if let Some(value) = json_usize(value, "n_rays")? {
        config.n_rays = value;
    }
    if let Some(value) = json_pair_usize(value, "grid")? {
        config.grid = value;
    }
    if let Some(value) =
        json_pair_usize(value, "patch_size")?.or(json_pair_usize(value, "train_patch_size")?)
    {
        config.patch_size = value;
    }
    if let Some(value) = json_usize(value, "batch_size")?.or(json_usize(value, "train_batch_size")?)
    {
        config.batch_size = value;
    }
    if let Some(value) = json_usize(value, "epochs")?.or(json_usize(value, "train_epochs")?) {
        config.epochs = value;
    }
    if let Some(value) =
        json_usize(value, "steps_per_epoch")?.or(json_usize(value, "train_steps_per_epoch")?)
    {
        config.steps_per_epoch = value;
    }
    if let Some(value) = json_usize(value, "validation_steps")? {
        config.validation_steps = value;
    }
    if let Some(value) =
        json_f64(value, "learning_rate")?.or(json_f64(value, "train_learning_rate")?)
    {
        config.learning_rate = value;
    }
    if let Some(value) =
        json_f32(value, "background_reg")?.or(json_f32(value, "train_background_reg")?)
    {
        config.background_reg = value;
    }
    if let Some(value) =
        json_f32(value, "foreground_probability")?.or(json_f32(value, "train_foreground_only")?)
    {
        config.foreground_probability = value;
    }
    if let Some(values) = json_vec_f32(value, "train_loss_weights")? {
        if values.len() == 2 {
            config.loss_prob_weight = values[0];
            config.loss_dist_weight = values[1];
        }
    }
    if let Some(value) = json_f32(value, "loss_prob_weight")? {
        config.loss_prob_weight = value;
    }
    if let Some(value) = json_f32(value, "loss_dist_weight")? {
        config.loss_dist_weight = value;
    }
    if let Some(value) =
        json_bool(value, "shape_completion")?.or(json_bool(value, "train_shape_completion")?)
    {
        config.shape_completion = value;
    }
    if let Some(value) =
        json_usize(value, "completion_crop")?.or(json_usize(value, "train_completion_crop")?)
    {
        config.completion_crop = value;
    }
    if let Some(value) = json_f32(value, "prob_threshold")? {
        config.prob_threshold = value;
    }
    if let Some(value) = json_f32(value, "nms_threshold")? {
        config.nms_threshold = value;
    }
    if let Some(reduce_lr) = json_get(value, "train_reduce_lr") {
        let factor = json_f64(reduce_lr, "factor")?.unwrap_or(0.5);
        let patience = json_usize(reduce_lr, "patience")?.unwrap_or(40);
        let min_delta = json_f32(reduce_lr, "min_delta")?.unwrap_or(0.0);
        config.lr_schedule = LearningRateSchedule2D::ReduceOnPlateau {
            factor,
            patience,
            min_delta,
            min_learning_rate: 0.0,
        };
    }
    config.validate()?;
    Ok(config)
}

fn apply_trained_model_config_overrides(
    config: &mut TrainingConfig2D,
    dict: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    let Some(dict) = dict else {
        return Ok(());
    };

    if let Some(value) = config_usize(Some(dict), "n_channel_in")? {
        ensure_structural_config_matches("n_channel_in", config.n_channel_in, value)?;
    }
    if let Some(value) = config_usize(Some(dict), "n_rays")? {
        ensure_structural_config_matches("n_rays", config.n_rays, value)?;
    }
    if let Some(value) = config_pair_usize(Some(dict), "grid")? {
        if value != config.grid {
            return Err(PyValueError::new_err(format!(
                "cannot override grid for a trained model: saved model uses {:?}, requested {:?}",
                config.grid, value
            )));
        }
    }
    if let Some(value) = config_f32(Some(dict), "prob_threshold")? {
        validate_threshold("prob_threshold", value)?;
        config.prob_threshold = value;
    }
    if let Some(value) = config_f32(Some(dict), "nms_threshold")? {
        validate_threshold("nms_threshold", value)?;
        config.nms_threshold = value;
    }
    apply_normalization_config(config, dict)?;
    Ok(())
}

fn ensure_structural_config_matches(name: &str, saved: usize, requested: usize) -> PyResult<()> {
    if saved == requested {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "cannot override {name} for a trained model: saved model uses {saved}, requested {requested}"
        )))
    }
}

fn validate_threshold(name: &str, value: f32) -> PyResult<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be in [0, 1]")))
    }
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
    if let Some(value) = config_pair_usize(Some(dict), "patch_size")?
        .or(config_pair_usize(Some(dict), "train_patch_size")?)
    {
        config.patch_size = value;
    }
    if let Some(value) =
        config_usize(Some(dict), "batch_size")?.or(config_usize(Some(dict), "train_batch_size")?)
    {
        config.batch_size = value;
    }
    if let Some(value) =
        config_usize(Some(dict), "epochs")?.or(config_usize(Some(dict), "train_epochs")?)
    {
        config.epochs = value;
    }
    if let Some(value) = config_usize(Some(dict), "steps_per_epoch")?
        .or(config_usize(Some(dict), "train_steps_per_epoch")?)
    {
        config.steps_per_epoch = value;
    }
    if let Some(value) = config_usize(Some(dict), "validation_steps")? {
        config.validation_steps = value;
    }
    if let Some(value) =
        config_f64(Some(dict), "learning_rate")?.or(config_f64(Some(dict), "train_learning_rate")?)
    {
        config.learning_rate = value;
    }
    if let Some(value) = config_f32(Some(dict), "weight_decay")? {
        config.weight_decay = value;
    }
    if let Some(value) = config_f32(Some(dict), "foreground_probability")?
        .or(config_f32(Some(dict), "train_foreground_only")?)
    {
        config.foreground_probability = value;
    }
    if let Some(value) = config_f32(Some(dict), "background_reg")?
        .or(config_f32(Some(dict), "train_background_reg")?)
    {
        config.background_reg = value;
    }
    if let Some(values) = config_vec_f32(Some(dict), "train_loss_weights")? {
        if values.len() != 2 {
            return Err(PyValueError::new_err(
                "train_loss_weights must have length 2",
            ));
        }
        config.loss_prob_weight = values[0];
        config.loss_dist_weight = values[1];
    }
    if let Some(value) = config_f32(Some(dict), "loss_prob_weight")? {
        config.loss_prob_weight = value;
    }
    if let Some(value) = config_f32(Some(dict), "loss_dist_weight")? {
        config.loss_dist_weight = value;
    }
    if let Some(value) = config_f32(Some(dict), "prob_threshold")? {
        validate_threshold("prob_threshold", value)?;
        config.prob_threshold = value;
    }
    if let Some(value) = config_f32(Some(dict), "nms_threshold")? {
        validate_threshold("nms_threshold", value)?;
        config.nms_threshold = value;
    }
    if let Some(value) = config_u64(Some(dict), "seed")? {
        config.seed = value;
    }
    if let Some(value) = config_usize(Some(dict), "validation_preview_count")? {
        config.validation_preview_count = value;
    }
    if let Some(value) = config_bool(Some(dict), "shape_completion")?
        .or(config_bool(Some(dict), "train_shape_completion")?)
    {
        config.shape_completion = value;
    }
    if let Some(value) = config_usize(Some(dict), "completion_crop")?
        .or(config_usize(Some(dict), "train_completion_crop")?)
    {
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

fn parse_bool_argument(value: Option<&Bound<'_, PyAny>>, name: &str) -> PyResult<bool> {
    let Some(value) = value else {
        return Ok(false);
    };
    if value.is_none() {
        return Ok(false);
    }
    if let Ok(value) = value.extract::<bool>() {
        return Ok(value);
    }
    if let Ok(value) = value.extract::<String>() {
        return match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Ok(true),
            "false" | "0" | "no" | "n" => Ok(false),
            _ => Err(PyValueError::new_err(format!(
                "{name} string value must be true or false"
            ))),
        };
    }
    Err(PyValueError::new_err(format!("{name} must be a bool")))
}

fn config_string(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<String>> {
    get_config_value(dict, key)
}

fn config_bool(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<bool>> {
    let Some(value) = get_config_item(dict, key)? else {
        return Ok(None);
    };
    parse_bool_argument(Some(&value), key).map(Some)
}

fn config_usize(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<usize>> {
    let Some(value) = get_config_item(dict, key)? else {
        return Ok(None);
    };
    py_any_to_usize(&value, key).map(Some)
}

fn config_u64(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<u64>> {
    let Some(value) = get_config_item(dict, key)? else {
        return Ok(None);
    };
    py_any_to_u64(&value, key).map(Some)
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
    let Some(value) = get_config_item(dict, key)? else {
        return Ok(None);
    };
    value.extract::<T>().map(Some).map_err(Into::into)
}

fn get_config_item<'py>(
    dict: Option<&Bound<'py, PyDict>>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(dict) = dict else {
        return Ok(None);
    };
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(value))
    }
}

fn py_any_to_usize(value: &Bound<'_, PyAny>, key: &str) -> PyResult<usize> {
    if let Ok(value) = value.extract::<usize>() {
        return Ok(value);
    }
    let value = value
        .extract::<f64>()
        .map_err(|_| PyValueError::new_err(format!("{key} must be an integer")))?;
    f64_to_usize(value, key)
}

fn py_any_to_u64(value: &Bound<'_, PyAny>, key: &str) -> PyResult<u64> {
    if let Ok(value) = value.extract::<u64>() {
        return Ok(value);
    }
    let value = value
        .extract::<f64>()
        .map_err(|_| PyValueError::new_err(format!("{key} must be an integer")))?;
    f64_to_u64(value, key)
}

fn f64_to_usize(value: f64, key: &str) -> PyResult<usize> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64 {
        Ok(value as usize)
    } else {
        Err(PyValueError::new_err(format!(
            "{key} must be a non-negative integer"
        )))
    }
}

fn f64_to_u64(value: f64, key: &str) -> PyResult<u64> {
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= u64::MAX as f64 {
        Ok(value as u64)
    } else {
        Err(PyValueError::new_err(format!(
            "{key} must be a non-negative integer"
        )))
    }
}

fn json_get<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    value.as_object()?.get(key).filter(|value| !value.is_null())
}

fn json_usize(value: &serde_json::Value, key: &str) -> Result<Option<usize>, StarDistTrainError> {
    let Some(value) = json_get(value, key) else {
        return Ok(None);
    };
    json_value_to_usize(value, key).map(Some)
}

fn json_f32(value: &serde_json::Value, key: &str) -> Result<Option<f32>, StarDistTrainError> {
    let Some(value) = json_get(value, key) else {
        return Ok(None);
    };
    json_value_to_f64(value, key).map(|value| Some(value as f32))
}

fn json_f64(value: &serde_json::Value, key: &str) -> Result<Option<f64>, StarDistTrainError> {
    let Some(value) = json_get(value, key) else {
        return Ok(None);
    };
    json_value_to_f64(value, key).map(Some)
}

fn json_bool(value: &serde_json::Value, key: &str) -> Result<Option<bool>, StarDistTrainError> {
    let Some(value) = json_get(value, key) else {
        return Ok(None);
    };
    if let Some(value) = value.as_bool() {
        return Ok(Some(value));
    }
    if let Some(value) = value.as_str() {
        return match value.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" => Ok(Some(true)),
            "false" | "0" | "no" | "n" => Ok(Some(false)),
            _ => Err(StarDistTrainError::InvalidConfig(format!(
                "{key} string value must be true or false"
            ))),
        };
    }
    Err(StarDistTrainError::InvalidConfig(format!(
        "{key} must be a boolean"
    )))
}

fn json_pair_usize(
    value: &serde_json::Value,
    key: &str,
) -> Result<Option<[usize; 2]>, StarDistTrainError> {
    let Some(values) = json_vec_usize(value, key)? else {
        return Ok(None);
    };
    if values.len() != 2 {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "{key} must have length 2"
        )));
    }
    Ok(Some([values[0], values[1]]))
}

fn json_vec_usize(
    value: &serde_json::Value,
    key: &str,
) -> Result<Option<Vec<usize>>, StarDistTrainError> {
    let Some(value) = json_get(value, key) else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "{key} must be a list of integers"
        )));
    };
    values
        .iter()
        .map(|value| json_value_to_usize(value, key))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn json_vec_f32(
    value: &serde_json::Value,
    key: &str,
) -> Result<Option<Vec<f32>>, StarDistTrainError> {
    let Some(value) = json_get(value, key) else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(StarDistTrainError::InvalidConfig(format!(
            "{key} must be a list of numbers"
        )));
    };
    values
        .iter()
        .map(|value| json_value_to_f64(value, key).map(|value| value as f32))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn json_value_to_usize(value: &serde_json::Value, key: &str) -> Result<usize, StarDistTrainError> {
    if let Some(value) = value.as_u64() {
        return usize::try_from(value).map_err(|_| {
            StarDistTrainError::InvalidConfig(format!("{key} is too large for this platform"))
        });
    }
    let value = json_value_to_f64(value, key)?;
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= usize::MAX as f64 {
        Ok(value as usize)
    } else {
        Err(StarDistTrainError::InvalidConfig(format!(
            "{key} must be a non-negative integer"
        )))
    }
}

fn json_value_to_f64(value: &serde_json::Value, key: &str) -> Result<f64, StarDistTrainError> {
    if let Some(value) = value.as_f64() {
        return Ok(value);
    }
    if let Some(value) = value.as_str() {
        return value
            .parse::<f64>()
            .map_err(|_| StarDistTrainError::InvalidConfig(format!("{key} must be a number")));
    }
    Err(StarDistTrainError::InvalidConfig(format!(
        "{key} must be a number"
    )))
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
    let Some(value) = get_config_item(dict, key)? else {
        return Ok(None);
    };
    if let Ok(values) = value.extract::<Vec<usize>>() {
        return Ok(Some(values));
    }
    let values = value
        .extract::<Vec<f64>>()
        .map_err(|_| PyValueError::new_err(format!("{key} must be a list of integers")))?;
    values
        .into_iter()
        .map(|value| f64_to_usize(value, key))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

fn config_vec_f32(dict: Option<&Bound<'_, PyDict>>, key: &str) -> PyResult<Option<Vec<f32>>> {
    let Some(value) = get_config_item(dict, key)? else {
        return Ok(None);
    };
    if let Ok(values) = value.extract::<Vec<f32>>() {
        return Ok(Some(values));
    }
    let values = value
        .extract::<Vec<f64>>()
        .map_err(|_| PyValueError::new_err(format!("{key} must be a list of numbers")))?;
    Ok(Some(values.into_iter().map(|value| value as f32).collect()))
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
