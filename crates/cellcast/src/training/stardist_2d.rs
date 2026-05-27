//! 2D StarDist training with Burn.
//!
//! This module is the first trainable Rust path for `cellcast`. It is designed
//! around the same data contract as StarDist, but it keeps tensors in Burn's
//! channel-first convention:
//!
//! - images: `[batch, channel, y, x]`
//! - object probabilities: `[batch, 1, y / 2, x / 2]`
//! - ray distances: `[batch, n_rays, y / 2, x / 2]`
//!
//! The conversion to StarDist's usual channel-last array format happens only
//! when calling the existing NMS/label-rendering code.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{CompactRecorder, RecorderError};
use burn::tensor::backend::{AutodiffBackend, Backend};
use ndarray::{Array2, Array3, Array4};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::models::stardist_2d::labels_from_prob_dist_2d;
use crate::networks::stardist::trainable_2d::{TrainableStarDist2D, TrainableStarDist2DConfig};

/// CPU backend with autodiff enabled for training.
pub type CpuTrainBackend = Autodiff<NdArray<f32, i32>>;
/// WebGPU backend with autodiff enabled for cross-vendor GPU training.
pub type WgpuTrainBackend = Autodiff<Wgpu<f32, i32>>;
/// CPU backend used after training for inference.
pub type CpuInferBackend = NdArray<f32, i32>;
/// WebGPU backend used after training for inference.
pub type WgpuInferBackend = Wgpu<f32, i32>;

/// One annotated 2D training image.
///
/// `image` must be channel-first `[channel, y, x]`. `labels` must be
/// `[y, x]`, with `0` as background and positive ids as object instances.
/// Negative labels follow the original StarDist convention: they mark pixels
/// where the probability loss should be ignored.
#[derive(Clone, Debug)]
pub struct TrainingSample2D {
    pub image: Array3<f32>,
    pub labels: Array2<i64>,
}

impl TrainingSample2D {
    /// Create a training sample and validate the spatial dimensions.
    pub fn new(image: Array3<f32>, labels: Array2<i64>) -> Result<Self, StarDistTrainError> {
        let image_shape = image.dim();
        let label_shape = labels.dim();
        if image_shape.1 != label_shape.0 || image_shape.2 != label_shape.1 {
            return Err(StarDistTrainError::Shape(format!(
                "image spatial shape ({}, {}) does not match label shape ({}, {})",
                image_shape.1, image_shape.2, label_shape.0, label_shape.1
            )));
        }
        Ok(Self { image, labels })
    }

    /// Convenience constructor for datasets that only contain non-negative
    /// instance ids.
    pub fn from_u64_labels(
        image: Array3<f32>,
        labels: Array2<u64>,
    ) -> Result<Self, StarDistTrainError> {
        let labels = labels.mapv(|label| label as i64);
        Self::new(image, labels)
    }
}

/// Data normalization policy.
///
/// Original StarDist training expects callers to pass already-normalized
/// images. `None` is therefore the default. Percentile normalization is kept as
/// an optional convenience for datasets loaded directly from files.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Normalization2D {
    None,
    Percentile { pmin: f32, pmax: f32 },
}

impl Default for Normalization2D {
    fn default() -> Self {
        Self::None
    }
}

/// CPU augmentation policy applied after crop extraction.
///
/// Spatial transforms are applied to image and labels together. Intensity
/// transforms are applied only to the image.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AugmentConfig2D {
    pub flip_y: bool,
    pub flip_x: bool,
    /// Random 90-degree rotations. Non-square patches skip this safely.
    pub rotate90: bool,
    /// Random multiplicative intensity factor in `[1 - v, 1 + v]`.
    pub intensity_scale: f32,
    /// Random additive intensity offset in `[-v, v]`.
    pub intensity_shift: f32,
    /// Gaussian noise standard deviation. Use `0.0` to disable.
    pub gaussian_noise_std: f32,
}

impl Default for AugmentConfig2D {
    fn default() -> Self {
        Self {
            flip_y: true,
            flip_x: true,
            rotate90: true,
            intensity_scale: 0.15,
            intensity_shift: 0.05,
            gaussian_noise_std: 0.01,
        }
    }
}

/// Training configuration for 2D single-class StarDist.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig2D {
    pub n_channel_in: usize,
    pub n_rays: usize,
    /// Prediction grid. `[1, 1]` predicts every pixel, `[2, 2]` predicts every
    /// second pixel and is faster/lighter.
    pub grid: [usize; 2],
    /// Patch size in `[y, x]`. Without shape completion, both dimensions must
    /// be divisible by 16. With shape completion, `patch_size - 2 *
    /// completion_crop` must be divisible by 16.
    pub patch_size: [usize; 2],
    pub batch_size: usize,
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub validation_steps: usize,
    pub learning_rate: f64,
    #[serde(default)]
    pub lr_schedule: LearningRateSchedule2D,
    pub weight_decay: f32,
    pub foreground_probability: f32,
    pub background_reg: f32,
    pub loss_prob_weight: f32,
    pub loss_dist_weight: f32,
    pub prob_threshold: f32,
    pub nms_threshold: f32,
    pub seed: u64,
    /// Number of validation samples to predict at the end of each epoch for
    /// progress displays. Keep this at `0` to avoid extra inference work.
    #[serde(default)]
    pub validation_preview_count: usize,
    #[serde(default)]
    pub shape_completion: bool,
    #[serde(default = "default_completion_crop_2d")]
    pub completion_crop: usize,
    pub normalization: Normalization2D,
    pub augment: AugmentConfig2D,
}

/// Learning-rate policy used by the training loop.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LearningRateSchedule2D {
    Constant,
    /// Match the original StarDist/Keras default: reduce LR when the monitored
    /// epoch loss plateaus.
    ReduceOnPlateau {
        factor: f64,
        patience: usize,
        min_delta: f32,
        min_learning_rate: f64,
    },
    /// Cosine annealing from `learning_rate` to `min_learning_rate`.
    CosineAnnealing {
        min_learning_rate: f64,
    },
    /// Polynomial decay from `learning_rate` to `min_learning_rate`.
    PolynomialDecay {
        power: f64,
        min_learning_rate: f64,
    },
}

impl Default for LearningRateSchedule2D {
    fn default() -> Self {
        Self::ReduceOnPlateau {
            factor: 0.5,
            patience: 40,
            min_delta: 0.0,
            min_learning_rate: 0.0,
        }
    }
}

fn default_completion_crop_2d() -> usize {
    32
}

/// JSON shape used by the original Python `stardist.models.Config2D`.
///
/// This is metadata compatibility, not Keras weight compatibility. Burn model
/// weights are still saved separately with Burn's recorder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonStarDist2DConfig {
    pub n_dim: usize,
    pub axes: String,
    pub n_channel_in: usize,
    pub n_channel_out: usize,
    pub train_checkpoint: String,
    pub train_checkpoint_last: String,
    pub train_checkpoint_epoch: String,
    pub n_rays: usize,
    pub grid: [usize; 2],
    pub backbone: String,
    pub n_classes: Option<usize>,
    pub unet_n_depth: usize,
    pub unet_kernel_size: [usize; 2],
    pub unet_n_filter_base: usize,
    pub unet_n_conv_per_depth: usize,
    pub unet_pool: [usize; 2],
    pub unet_activation: String,
    pub unet_last_activation: String,
    pub unet_batch_norm: bool,
    pub unet_dropout: f32,
    pub unet_expansion: usize,
    pub unet_prefix: String,
    pub net_conv_after_unet: usize,
    pub net_input_shape: [Option<usize>; 3],
    pub net_mask_shape: [Option<usize>; 3],
    pub train_shape_completion: bool,
    pub train_completion_crop: usize,
    pub train_patch_size: [usize; 2],
    pub train_background_reg: f32,
    pub train_foreground_only: f32,
    pub train_sample_cache: bool,
    pub train_dist_loss: String,
    pub train_loss_weights: [f32; 2],
    pub train_class_weights: [f32; 2],
    pub train_epochs: usize,
    pub train_steps_per_epoch: usize,
    pub train_learning_rate: f64,
    pub train_batch_size: usize,
    pub train_n_val_patches: Option<usize>,
    pub train_tensorboard: bool,
    pub train_reduce_lr: PythonReduceLrConfig,
    pub use_gpu: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PythonReduceLrConfig {
    pub factor: f32,
    pub patience: usize,
    pub min_delta: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PythonThresholds {
    pub prob: f32,
    pub nms: f32,
}

/// Candidate grid used when optimizing prediction thresholds on validation data.
#[derive(Clone, Debug)]
pub struct ThresholdOptimizationConfig2D {
    pub prob_thresholds: Vec<f32>,
    pub nms_thresholds: Vec<f32>,
    pub iou_thresholds: Vec<f32>,
    pub max_samples: Option<usize>,
}

impl Default for ThresholdOptimizationConfig2D {
    fn default() -> Self {
        Self {
            prob_thresholds: (1..=18).map(|i| i as f32 * 0.05).collect(),
            nms_thresholds: vec![0.3, 0.4, 0.5],
            iou_thresholds: vec![0.3, 0.5, 0.7],
            max_samples: None,
        }
    }
}

/// Raw dense StarDist2D prediction before NMS.
#[derive(Clone, Debug)]
pub struct Prediction2D {
    pub prob: Array2<f32>,
    pub dist: Array3<f32>,
}

impl Default for TrainingConfig2D {
    fn default() -> Self {
        Self {
            n_channel_in: 1,
            n_rays: 32,
            grid: [1, 1],
            patch_size: [256, 256],
            batch_size: 4,
            epochs: 400,
            steps_per_epoch: 100,
            validation_steps: 10,
            learning_rate: 3e-4,
            lr_schedule: LearningRateSchedule2D::default(),
            weight_decay: 0.0,
            foreground_probability: 0.9,
            background_reg: 1e-4,
            loss_prob_weight: 1.0,
            loss_dist_weight: 0.2,
            prob_threshold: 0.5,
            nms_threshold: 0.4,
            seed: 42,
            validation_preview_count: 0,
            shape_completion: false,
            completion_crop: default_completion_crop_2d(),
            normalization: Normalization2D::default(),
            augment: AugmentConfig2D::default(),
        }
    }
}

impl TrainingConfig2D {
    pub fn validate(&self) -> Result<(), StarDistTrainError> {
        if self.n_channel_in == 0 {
            return Err(StarDistTrainError::InvalidConfig(
                "n_channel_in must be positive".to_string(),
            ));
        }
        if self.n_rays < 3 {
            return Err(StarDistTrainError::InvalidConfig(
                "n_rays must be at least 3".to_string(),
            ));
        }
        if self.grid != [1, 1] && self.grid != [2, 2] {
            return Err(StarDistTrainError::InvalidConfig(
                "the current trainable StarDist2D architecture supports grid [1, 1] and [2, 2]"
                    .to_string(),
            ));
        }
        if self.patch_size[0] == 0 || self.patch_size[1] == 0 {
            return Err(StarDistTrainError::InvalidConfig(
                "patch_size entries must be positive".to_string(),
            ));
        }
        if !self.shape_completion && (self.patch_size[0] % 16 != 0 || self.patch_size[1] % 16 != 0)
        {
            return Err(StarDistTrainError::InvalidConfig(
                "patch_size must be divisible by 16 for the 4-level U-Net".to_string(),
            ));
        }
        if self.shape_completion {
            let crop = self.completion_crop;
            if crop == 0 {
                return Err(StarDistTrainError::InvalidConfig(
                    "completion_crop must be positive when shape_completion is enabled".to_string(),
                ));
            }
            if 2 * crop >= self.patch_size[0] || 2 * crop >= self.patch_size[1] {
                return Err(StarDistTrainError::InvalidConfig(
                    "shape completion requires patch_size to be larger than 2 * completion_crop"
                        .to_string(),
                ));
            }
            if crop % self.grid[0] != 0 || crop % self.grid[1] != 0 {
                return Err(StarDistTrainError::InvalidConfig(
                    "completion_crop must be divisible by all grid values".to_string(),
                ));
            }
            let input_h = self.patch_size[0] - 2 * crop;
            let input_w = self.patch_size[1] - 2 * crop;
            if input_h % 16 != 0 || input_w % 16 != 0 {
                return Err(StarDistTrainError::InvalidConfig(
                    "patch_size - 2 * completion_crop must be divisible by 16".to_string(),
                ));
            }
        }
        if self.batch_size == 0 {
            return Err(StarDistTrainError::InvalidConfig(
                "batch_size must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.foreground_probability) {
            return Err(StarDistTrainError::InvalidConfig(
                "foreground_probability must be in [0, 1]".to_string(),
            ));
        }
        match self.lr_schedule {
            LearningRateSchedule2D::Constant => {}
            LearningRateSchedule2D::ReduceOnPlateau {
                factor,
                patience,
                min_learning_rate,
                ..
            } => {
                if !(0.0..1.0).contains(&factor) {
                    return Err(StarDistTrainError::InvalidConfig(
                        "ReduceOnPlateau factor must be in (0, 1)".to_string(),
                    ));
                }
                if patience == 0 {
                    return Err(StarDistTrainError::InvalidConfig(
                        "ReduceOnPlateau patience must be positive".to_string(),
                    ));
                }
                if min_learning_rate < 0.0 || min_learning_rate > self.learning_rate {
                    return Err(StarDistTrainError::InvalidConfig(
                        "ReduceOnPlateau min_learning_rate must be between 0 and learning_rate"
                            .to_string(),
                    ));
                }
            }
            LearningRateSchedule2D::CosineAnnealing { min_learning_rate } => {
                if min_learning_rate < 0.0 || min_learning_rate > self.learning_rate {
                    return Err(StarDistTrainError::InvalidConfig(
                        "CosineAnnealing min_learning_rate must be between 0 and learning_rate"
                            .to_string(),
                    ));
                }
            }
            LearningRateSchedule2D::PolynomialDecay {
                power,
                min_learning_rate,
            } => {
                if power <= 0.0 {
                    return Err(StarDistTrainError::InvalidConfig(
                        "PolynomialDecay power must be positive".to_string(),
                    ));
                }
                if min_learning_rate < 0.0 || min_learning_rate > self.learning_rate {
                    return Err(StarDistTrainError::InvalidConfig(
                        "PolynomialDecay min_learning_rate must be between 0 and learning_rate"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Convert this Rust training config into a Python StarDist `Config2D`
    /// compatible JSON object.
    pub fn to_python_config(&self) -> PythonStarDist2DConfig {
        PythonStarDist2DConfig {
            n_dim: 2,
            axes: "YXC".to_string(),
            n_channel_in: self.n_channel_in,
            n_channel_out: 1 + self.n_rays,
            train_checkpoint: "weights_best.h5".to_string(),
            train_checkpoint_last: "weights_last.h5".to_string(),
            train_checkpoint_epoch: "weights_now.h5".to_string(),
            n_rays: self.n_rays,
            grid: self.grid,
            backbone: "unet".to_string(),
            n_classes: None,
            unet_n_depth: 3,
            unet_kernel_size: [3, 3],
            unet_n_filter_base: 32,
            unet_n_conv_per_depth: 2,
            unet_pool: [2, 2],
            unet_activation: "relu".to_string(),
            unet_last_activation: "relu".to_string(),
            unet_batch_norm: false,
            unet_dropout: 0.0,
            unet_expansion: 2,
            unet_prefix: String::new(),
            net_conv_after_unet: 128,
            net_input_shape: [None, None, Some(self.n_channel_in)],
            net_mask_shape: [None, None, Some(1)],
            train_shape_completion: self.shape_completion,
            train_completion_crop: self.completion_crop,
            train_patch_size: self.patch_size,
            train_background_reg: self.background_reg,
            train_foreground_only: self.foreground_probability,
            train_sample_cache: true,
            train_dist_loss: "mae".to_string(),
            train_loss_weights: [self.loss_prob_weight, self.loss_dist_weight],
            train_class_weights: [1.0, 1.0],
            train_epochs: self.epochs,
            train_steps_per_epoch: self.steps_per_epoch,
            train_learning_rate: self.learning_rate,
            train_batch_size: self.batch_size,
            train_n_val_patches: None,
            train_tensorboard: true,
            train_reduce_lr: self.python_reduce_lr_config(),
            use_gpu: false,
        }
    }

    fn python_reduce_lr_config(&self) -> PythonReduceLrConfig {
        match self.lr_schedule {
            LearningRateSchedule2D::ReduceOnPlateau {
                factor,
                patience,
                min_delta,
                ..
            } => PythonReduceLrConfig {
                factor: factor as f32,
                patience,
                min_delta,
            },
            _ => PythonReduceLrConfig {
                factor: 0.5,
                patience: 40,
                min_delta: 0.0,
            },
        }
    }

    pub fn python_thresholds(&self) -> PythonThresholds {
        PythonThresholds {
            prob: self.prob_threshold,
            nms: self.nms_threshold,
        }
    }
}

/// Mean losses for one epoch.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub learning_rate: f64,
    pub train_total: f32,
    pub train_prob: f32,
    pub train_dist: f32,
    pub valid_total: Option<f32>,
    pub valid_prob: Option<f32>,
    pub valid_dist: Option<f32>,
}

/// Static information emitted before the first training step.
#[derive(Clone, Copy, Debug)]
pub struct TrainingPlan2D {
    pub epochs: usize,
    pub steps_per_epoch: usize,
    pub validation_steps: usize,
    pub total_steps: usize,
    pub train_samples: usize,
    pub valid_samples: usize,
    pub batch_size: usize,
    pub patch_size: [usize; 2],
    pub grid: [usize; 2],
    pub n_rays: usize,
    pub validation_preview_count: usize,
}

/// Losses observed after one optimizer step.
#[derive(Clone, Copy, Debug)]
pub struct StepMetrics2D {
    pub epoch: usize,
    pub step: usize,
    pub global_step: usize,
    pub learning_rate: f64,
    pub total: f32,
    pub prob: f32,
    pub dist: f32,
}

/// Optional validation output that can be displayed by a UI while training.
#[derive(Clone, Debug)]
pub struct ValidationPreview2D {
    pub sample_index: usize,
    pub image: Array3<f32>,
    pub labels: Array2<i64>,
    pub prediction: Array2<u64>,
    pub prob: Array2<f32>,
}

/// Event emitted after each epoch, after the optional validation pass.
#[derive(Clone, Debug)]
pub struct EpochEndEvent2D {
    pub metrics: EpochMetrics,
    pub validation_ran: bool,
    pub previews: Vec<ValidationPreview2D>,
}

/// Observer interface for training progress.
///
/// The default methods are no-ops, so callers can implement only the callbacks
/// they need. The Python package implements this trait by converting these
/// events into Python dictionaries and calling Python callables.
pub trait TrainingCallbacks2D {
    fn on_train_begin(&mut self, _plan: &TrainingPlan2D) -> Result<(), StarDistTrainError> {
        Ok(())
    }

    fn on_step_end(&mut self, _metrics: &StepMetrics2D) -> Result<(), StarDistTrainError> {
        Ok(())
    }

    fn on_epoch_end(&mut self, _event: &EpochEndEvent2D) -> Result<(), StarDistTrainError> {
        Ok(())
    }
}

/// Default callback implementation used when the caller does not observe
/// training.
#[derive(Default)]
pub struct NoOpTrainingCallbacks2D;

impl TrainingCallbacks2D for NoOpTrainingCallbacks2D {}

/// Result returned by `train_stardist_2d`.
#[derive(Debug)]
pub struct TrainingResult2D<B: Backend> {
    pub model: TrainableStarDist2D<B>,
    pub history: Vec<EpochMetrics>,
    pub config: TrainingConfig2D,
}

/// Errors returned by the Rust StarDist trainer.
#[derive(Debug)]
pub enum StarDistTrainError {
    Callback(String),
    Dataset(String),
    EmptyDataset,
    Image {
        path: PathBuf,
        source: image::ImageError,
    },
    InvalidConfig(String),
    Io(std::io::Error),
    Recorder(RecorderError),
    Serde(serde_json::Error),
    Shape(String),
}

impl Display for StarDistTrainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Callback(msg) => write!(f, "training callback error: {msg}"),
            Self::Dataset(msg) => write!(f, "dataset error: {msg}"),
            Self::EmptyDataset => write!(f, "training dataset is empty"),
            Self::Image { path, source } => {
                write!(f, "could not decode image {}: {source}", path.display())
            }
            Self::InvalidConfig(msg) => write!(f, "invalid training config: {msg}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Recorder(err) => write!(f, "model record error: {err}"),
            Self::Serde(err) => write!(f, "serialization error: {err}"),
            Self::Shape(msg) => write!(f, "shape error: {msg}"),
        }
    }
}

impl Error for StarDistTrainError {}

impl From<std::io::Error> for StarDistTrainError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<RecorderError> for StarDistTrainError {
    fn from(value: RecorderError) -> Self {
        Self::Recorder(value)
    }
}

impl From<serde_json::Error> for StarDistTrainError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serde(value)
    }
}

struct Batch2D<B: Backend> {
    image: Tensor<B, 4>,
    prob: Tensor<B, 4>,
    prob_mask: Tensor<B, 4>,
    dist: Tensor<B, 4>,
    dist_mask: Tensor<B, 4>,
}

struct LossTensors<B: Backend> {
    total: Tensor<B, 1>,
    prob: Tensor<B, 1>,
    dist: Tensor<B, 1>,
}

#[derive(Clone, Copy, Debug)]
struct LearningRateState2D {
    current: f64,
    best_loss: f32,
    plateau_epochs: usize,
}

impl LearningRateState2D {
    fn new(initial: f64) -> Self {
        Self {
            current: initial,
            best_loss: f32::INFINITY,
            plateau_epochs: 0,
        }
    }

    fn learning_rate(&self, config: &TrainingConfig2D, epoch: usize) -> f64 {
        match config.lr_schedule {
            LearningRateSchedule2D::Constant | LearningRateSchedule2D::ReduceOnPlateau { .. } => {
                self.current
            }
            LearningRateSchedule2D::CosineAnnealing { min_learning_rate } => {
                let progress = epoch_progress(epoch, config.epochs);
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
                min_learning_rate + (config.learning_rate - min_learning_rate) * cosine
            }
            LearningRateSchedule2D::PolynomialDecay {
                power,
                min_learning_rate,
            } => {
                let progress = epoch_progress(epoch, config.epochs);
                let decay = (1.0 - progress).max(0.0).powf(power);
                min_learning_rate + (config.learning_rate - min_learning_rate) * decay
            }
        }
    }

    fn update_after_epoch(&mut self, schedule: &LearningRateSchedule2D, monitored_loss: f32) {
        if let LearningRateSchedule2D::ReduceOnPlateau {
            factor,
            patience,
            min_delta,
            min_learning_rate,
        } = *schedule
        {
            if monitored_loss < self.best_loss - min_delta {
                self.best_loss = monitored_loss;
                self.plateau_epochs = 0;
                return;
            }

            self.plateau_epochs += 1;
            if self.plateau_epochs >= patience {
                self.current = (self.current * factor).max(min_learning_rate);
                self.plateau_epochs = 0;
            }
        }
    }
}

fn epoch_progress(epoch: usize, epochs: usize) -> f64 {
    if epochs <= 1 {
        1.0
    } else {
        (epoch.saturating_sub(1) as f64) / ((epochs - 1) as f64)
    }
}

/// Train a 2D StarDist model with any Burn autodiff backend.
///
/// `B` is the Burn backend type. For CPU training it is `CpuTrainBackend`; for
/// WebGPU training it is `WgpuTrainBackend`. The same training code works with
/// both because Burn exposes tensors, gradients, and optimizers through backend
/// traits.
pub fn train_stardist_2d<B>(
    // Borrow the training samples as a slice. The function reads from this
    // dataset but does not take ownership of it, so the caller can still use the
    // samples after training returns.
    train_samples: &[TrainingSample2D],
    // Validation is optional. `None` means train without validation metrics.
    // `Some(&samples)` means compute validation loss at the end of each epoch.
    valid_samples: Option<&[TrainingSample2D]>,
    // Take ownership of the config. The result stores the same config, so the
    // caller gets a record of exactly what was used for training.
    config: TrainingConfig2D,
    // Burn device where tensors and model parameters live. Depending on `B`,
    // this can be CPU or a GPU device.
    device: B::Device,
    // Return either a trained model plus metrics, or a structured training
    // error.
    //
    // The returned model uses `B::InnerBackend`, not `B`. `B` is an autodiff
    // backend used for training; `B::InnerBackend` is the plain inference
    // backend underneath it. After training we do not need gradient tracking
    // anymore.
) -> Result<TrainingResult2D<B::InnerBackend>, StarDistTrainError>
where
    // This function only accepts Burn backends that support automatic
    // differentiation. `FloatElem = f32` fixes tensor floating-point values to
    // 32-bit floats, which is what the model and losses use. `IntElem = i32`
    // fixes integer tensors to 32-bit ints.
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
{
    let mut callbacks = NoOpTrainingCallbacks2D;
    train_stardist_2d_with_callbacks::<B, _>(
        train_samples,
        valid_samples,
        config,
        device,
        &mut callbacks,
    )
}

/// Train a 2D StarDist model and report progress through callbacks.
pub fn train_stardist_2d_with_callbacks<B, C>(
    train_samples: &[TrainingSample2D],
    valid_samples: Option<&[TrainingSample2D]>,
    config: TrainingConfig2D,
    device: B::Device,
    callbacks: &mut C,
) -> Result<TrainingResult2D<B::InnerBackend>, StarDistTrainError>
where
    B: AutodiffBackend<FloatElem = f32, IntElem = i32>,
    C: TrainingCallbacks2D,
{
    // Check that the training configuration is internally consistent before we
    // allocate the model or start consuming time.
    config.validate()?;

    // Check that every training image has the expected channel count and is big
    // enough for the requested patch size.
    validate_samples(train_samples, &config)?;

    // If validation data was provided, validate it with the same rules.
    if let Some(valid) = valid_samples {
        validate_samples(valid, &config)?;
    }

    let valid_sample_count = valid_samples.map(|samples| samples.len()).unwrap_or(0);
    callbacks.on_train_begin(&TrainingPlan2D {
        epochs: config.epochs,
        steps_per_epoch: config.steps_per_epoch,
        validation_steps: config.validation_steps,
        total_steps: config.epochs.saturating_mul(config.steps_per_epoch),
        train_samples: train_samples.len(),
        valid_samples: valid_sample_count,
        batch_size: config.batch_size,
        patch_size: config.patch_size,
        grid: config.grid,
        n_rays: config.n_rays,
        validation_preview_count: config.validation_preview_count,
    })?;

    // Seed Burn's backend-level random generator. This affects random
    // operations controlled by the backend, such as model parameter
    // initialization.
    B::seed(&device, config.seed);

    // Build the model architecture configuration from the training config. This
    // is separate from `TrainingConfig2D`: it only contains model-shape choices.
    let model_config =
        TrainableStarDist2DConfig::new(config.n_channel_in, config.n_rays, config.grid);

    // Initialize the trainable Burn model on the chosen backend and device. The
    // `::<B>` syntax tells Rust exactly which backend type to use here.
    let mut model = model_config.init::<B>(&device);

    // Create the AdamW optimizer. `mut` is required because the optimizer keeps
    // internal state, such as first and second moment estimates.
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init();

    // Create a normal Rust random-number generator for CPU-side sampling:
    // choosing images, crop positions, and augmentations.
    let mut rng = StdRng::seed_from_u64(config.seed);
    // Validation should measure the same held-out patches every epoch, matching
    // original StarDist's fixed `data_val`. We keep a separate deterministic
    // seed so validation sampling does not depend on how many training batches
    // have already been drawn.
    let validation_seed = config.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);

    // Track the current learning rate and scheduler state. For example,
    // ReduceLROnPlateau needs to remember the best loss seen so far.
    let mut lr_state = LearningRateState2D::new(config.learning_rate);

    // Pre-allocate the metrics vector. This avoids repeated reallocations while
    // pushing one `EpochMetrics` value per epoch.
    let mut history = Vec::with_capacity(config.epochs);
    let mut global_step = 0usize;

    // Rust ranges are explicit. `1..=config.epochs` includes both endpoints, so
    // epoch numbers are 1-based: 1, 2, ..., config.epochs.
    for epoch in 1..=config.epochs {
        // Ask the scheduler what learning rate should be used for this epoch.
        let learning_rate = lr_state.learning_rate(&config, epoch);

        // Accumulators for the average training losses over this epoch.
        let mut train_total = 0.0;
        let mut train_prob = 0.0;
        let mut train_dist = 0.0;

        // A step is one random mini-batch update. `steps_per_epoch` controls how
        // many random batches are drawn before we call the epoch finished.
        for step in 1..=config.steps_per_epoch {
            // Build a training batch as Burn tensors on backend `B`. The final
            // `true` enables training-time behavior such as augmentation and
            // foreground-biased random crops.
            let batch = build_batch_2d::<B>(train_samples, &config, &device, &mut rng, true)?;

            // Run the forward pass. StarDist has two heads: object probability
            // and radial distances.
            let (prob_pred, dist_pred) = model.forward(batch.image);

            // Compare predictions with the generated StarDist targets. The loss
            // object keeps Burn tensors so `total.backward()` can compute
            // gradients through the computation graph.
            let losses = stardist_loss(
                prob_pred,
                dist_pred,
                batch.prob,
                batch.prob_mask,
                batch.dist,
                batch.dist_mask,
                &config,
            );

            // Convert scalar tensors to Rust `f32` values for logging. We clone
            // because `into_scalar()` consumes the tensor, and `losses.total`
            // still needs to be used for backpropagation below.
            let step_total = losses.total.clone().into_scalar();
            let step_prob = losses.prob.clone().into_scalar();
            let step_dist = losses.dist.clone().into_scalar();
            train_total += step_total;
            train_prob += step_prob;
            train_dist += step_dist;

            // Ask Burn autodiff to compute gradients of every trainable
            // parameter with respect to the total loss.
            let grads = losses.total.backward();

            // Convert raw autodiff gradients into the parameter-keyed structure
            // expected by Burn optimizers.
            let grads = GradientsParams::from_grads(grads, &model);

            // Apply one AdamW update. Burn optimizers return the updated model,
            // so we assign it back to `model`.
            model = optim.step(learning_rate, model, grads);

            global_step += 1;
            callbacks.on_step_end(&StepMetrics2D {
                epoch,
                step,
                global_step,
                learning_rate,
                total: step_total,
                prob: step_prob,
                dist: step_dist,
            })?;
        }

        // Avoid division by zero if a config somehow has zero steps. Validation
        // already rejects invalid configs, but this keeps the averaging robust.
        let denom = config.steps_per_epoch.max(1) as f32;

        // Store average training metrics for this epoch. Validation fields start
        // as `None` and are filled only if validation runs below.
        let mut metrics = EpochMetrics {
            epoch,
            learning_rate,
            train_total: train_total / denom,
            train_prob: train_prob / denom,
            train_dist: train_dist / denom,
            valid_total: None,
            valid_prob: None,
            valid_dist: None,
        };

        let mut validation_ran = false;
        let mut previews = Vec::new();

        // Pattern-match the optional validation dataset. If it is `None`, the
        // validation-loss and validation-preview blocks are skipped.
        if let Some(valid) = valid_samples {
            if !valid.is_empty() {
                // Convert the autodiff model into a plain inference model. This
                // disables gradient tracking for validation and previews.
                let valid_model: TrainableStarDist2D<<B as AutodiffBackend>::InnerBackend> = model.valid();

                // Only run validation loss if the config asks for at least one
                // validation batch.
                if config.validation_steps > 0 {
                    validation_ran = true;

                    // Accumulators for average validation losses.
                    let mut valid_total = 0.0;
                    let mut valid_prob = 0.0;
                    let mut valid_dist = 0.0;
                    let mut validation_rng = StdRng::seed_from_u64(validation_seed);

                    // Draw the same validation batches every epoch. Validation
                    // uses the same target generation code but disables
                    // training augmentations.
                    for _ in 0..config.validation_steps {
                        // Build tensors with the inner non-autodiff backend
                        // because validation does not call backward.
                        let batch: Batch2D<<B as AutodiffBackend>::InnerBackend> = build_batch_2d::<B::InnerBackend>(
                            valid, &config, &device, &mut validation_rng, false,
                        )?;

                        // Forward pass for validation.
                        let (prob_pred, dist_pred) = valid_model.forward(batch.image);

                        // Compute the same StarDist loss, but only for
                        // measurement. No gradients are computed from this loss.
                        let losses: LossTensors<<B as AutodiffBackend>::InnerBackend> = stardist_loss(
                            prob_pred,
                            dist_pred,
                            batch.prob,
                            batch.prob_mask,
                            batch.dist,
                            batch.dist_mask,
                            &config,
                        );

                        // Validation losses are consumed immediately because we
                        // do not need them for backpropagation.
                        valid_total += losses.total.into_scalar();
                        valid_prob += losses.prob.into_scalar();
                        valid_dist += losses.dist.into_scalar();
                    }

                    // Average validation losses over the requested number of
                    // validation batches.
                    let denom = config.validation_steps as f32;
                    metrics.valid_total = Some(valid_total / denom);
                    metrics.valid_prob = Some(valid_prob / denom);
                    metrics.valid_dist = Some(valid_dist / denom);
                }

                if config.validation_preview_count > 0 {
                    previews = validation_previews_2d::<B::InnerBackend>(
                        &valid_model,
                        valid,
                        &config,
                        &device,
                    )?;
                }
            }
        }

        // Learning-rate schedulers usually monitor validation loss when it is
        // available. If there is no validation loss, fall back to training loss.
        let monitored_loss = metrics.valid_total.unwrap_or(metrics.train_total);

        // Update scheduler state after the epoch. For ReduceLROnPlateau this may
        // reduce the future learning rate if the monitored loss has stalled.
        lr_state.update_after_epoch(&config.lr_schedule, monitored_loss);

        callbacks.on_epoch_end(&EpochEndEvent2D {
            metrics,
            validation_ran,
            previews,
        })?;

        // Save this epoch's metrics into the returned training history.
        history.push(metrics);
    }

    // Return the trained model as an inference model, together with all epoch
    // metrics and the config used to train it.
    Ok(TrainingResult2D {
        model: model.valid(),
        history,
        config,
    })
}

/// Convenience CPU entry point.
pub fn train_stardist_2d_cpu(
    train_samples: &[TrainingSample2D],
    valid_samples: Option<&[TrainingSample2D]>,
    config: TrainingConfig2D,
) -> Result<TrainingResult2D<CpuInferBackend>, StarDistTrainError> {
    let device = Default::default();
    train_stardist_2d::<CpuTrainBackend>(train_samples, valid_samples, config, device)
}

/// Convenience WebGPU entry point for NVIDIA, AMD, Intel, and Apple GPUs where
/// WGPU is available.
pub fn train_stardist_2d_wgpu(
    train_samples: &[TrainingSample2D],
    valid_samples: Option<&[TrainingSample2D]>,
    config: TrainingConfig2D,
) -> Result<TrainingResult2D<WgpuInferBackend>, StarDistTrainError> {
    let device = Default::default();
    train_stardist_2d::<WgpuTrainBackend>(train_samples, valid_samples, config, device)
}

fn validation_previews_2d<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    samples: &[TrainingSample2D],
    config: &TrainingConfig2D,
    device: &B::Device,
) -> Result<Vec<ValidationPreview2D>, StarDistTrainError> {
    let count = config.validation_preview_count.min(samples.len());
    let mut previews = Vec::with_capacity(count);

    for (sample_index, sample) in samples.iter().take(count).enumerate() {
        let (_, height, width) = sample.image.dim();
        let raw = predict_stardist_2d_raw(model, &sample.image, config, device)?;
        let prediction = labels_from_prob_dist_2d(
            raw.prob.clone(),
            raw.dist,
            config.prob_threshold,
            config.nms_threshold,
            config.grid,
            (height, width),
        );
        previews.push(ValidationPreview2D {
            sample_index,
            image: sample.image.clone(),
            labels: sample.labels.clone(),
            prediction,
            prob: raw.prob,
        });
    }

    Ok(previews)
}

/// Run trained-model inference and return instance labels.
///
/// `image` is channel-first `[channel, y, x]`. The model output is converted to
/// `[y, x, rays]` only at the NMS boundary.
pub fn predict_stardist_2d<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    image: &Array3<f32>,
    config: &TrainingConfig2D,
    device: &B::Device,
) -> Result<Array2<u64>, StarDistTrainError> {
    predict_stardist_2d_with_thresholds(
        model,
        image,
        config,
        config.prob_threshold,
        config.nms_threshold,
        device,
    )
}

/// Run trained-model inference with explicit probability and NMS thresholds.
pub fn predict_stardist_2d_with_thresholds<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    image: &Array3<f32>,
    config: &TrainingConfig2D,
    prob_threshold: f32,
    nms_threshold: f32,
    device: &B::Device,
) -> Result<Array2<u64>, StarDistTrainError> {
    let prediction = predict_stardist_2d_raw(model, image, config, device)?;
    let (_, height, width) = image.dim();
    Ok(labels_from_prob_dist_2d(
        prediction.prob,
        prediction.dist,
        prob_threshold,
        nms_threshold,
        config.grid,
        (height, width),
    ))
}

/// Run trained-model inference for a batch of same-size channel-first images.
///
/// The neural network forward pass is batched as `[batch, channel, y, x]`.
/// StarDist NMS and label rendering are still applied independently per image.
pub fn predict_stardist_2d_batch_with_thresholds<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    images: &Array4<f32>,
    config: &TrainingConfig2D,
    prob_threshold: f32,
    nms_threshold: f32,
    device: &B::Device,
) -> Result<Array3<u64>, StarDistTrainError> {
    let predictions = predict_stardist_2d_batch_raw(model, images, config, device)?;
    let (batch_size, _, height, width) = images.dim();
    let mut labels = Array3::<u64>::zeros((batch_size, height, width));
    for (batch_index, prediction) in predictions.into_iter().enumerate() {
        let label_image = labels_from_prob_dist_2d(
            prediction.prob,
            prediction.dist,
            prob_threshold,
            nms_threshold,
            config.grid,
            (height, width),
        );
        for y in 0..height {
            for x in 0..width {
                labels[[batch_index, y, x]] = label_image[[y, x]];
            }
        }
    }
    Ok(labels)
}

/// Run trained-model inference and return dense probability and distance maps.
pub fn predict_stardist_2d_raw<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    image: &Array3<f32>,
    config: &TrainingConfig2D,
    device: &B::Device,
) -> Result<Prediction2D, StarDistTrainError> {
    config.validate()?;
    let (channels, _, _) = image.dim();
    if channels != config.n_channel_in {
        return Err(StarDistTrainError::Shape(format!(
            "expected {} channels, got {}",
            config.n_channel_in, channels
        )));
    }

    let normalized = normalize_image(image, config.normalization);
    let padded = reflect_pad_to_divisible(&normalized, 16);
    let (_, padded_h, padded_w) = padded.dim();
    let (raw, _) = padded.into_raw_vec_and_offset();
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(raw, [1, channels, padded_h, padded_w]),
        device,
    );
    let (prob, dist) = model.forward(input);
    let prob_dims = prob.dims();
    let dist_dims = dist.dims();
    let out_h = prob_dims[2];
    let out_w = prob_dims[3];
    if dist_dims[1] != config.n_rays || dist_dims[2] != out_h || dist_dims[3] != out_w {
        return Err(StarDistTrainError::Shape(format!(
            "prob/dist output shapes are incompatible: {:?} vs {:?}",
            prob_dims, dist_dims
        )));
    }

    let prob_vec = prob.into_data().into_vec().unwrap();
    let dist_vec = dist.into_data().into_vec().unwrap();
    let mut prob_arr = Array2::<f32>::zeros((out_h, out_w));
    let mut dist_arr = Array3::<f32>::zeros((out_h, out_w, config.n_rays));

    for y in 0..out_h {
        for x in 0..out_w {
            prob_arr[[y, x]] = prob_vec[y * out_w + x];
            for ray in 0..config.n_rays {
                let idx = ((ray * out_h) + y) * out_w + x;
                dist_arr[[y, x, ray]] = dist_vec[idx];
            }
        }
    }

    Ok(Prediction2D {
        prob: prob_arr,
        dist: dist_arr,
    })
}

/// Run batched trained-model inference and return dense probability/distance
/// maps for each image before NMS.
pub fn predict_stardist_2d_batch_raw<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    images: &Array4<f32>,
    config: &TrainingConfig2D,
    device: &B::Device,
) -> Result<Vec<Prediction2D>, StarDistTrainError> {
    config.validate()?;
    let (batch_size, channels, height, width) = images.dim();
    if batch_size == 0 {
        return Err(StarDistTrainError::Shape(
            "batch dimension must be positive".to_string(),
        ));
    }
    if height == 0 || width == 0 {
        return Err(StarDistTrainError::Shape(
            "image height and width must be positive".to_string(),
        ));
    }
    if channels != config.n_channel_in {
        return Err(StarDistTrainError::Shape(format!(
            "expected {} channels, got {}",
            config.n_channel_in, channels
        )));
    }

    let padded_h = height + (16 - (height % 16)) % 16;
    let padded_w = width + (16 - (width % 16)) % 16;
    let mut input_batch = Array4::<f32>::zeros((batch_size, channels, padded_h, padded_w));
    for batch_index in 0..batch_size {
        let mut image = Array3::<f32>::zeros((channels, height, width));
        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    image[[c, y, x]] = images[[batch_index, c, y, x]];
                }
            }
        }
        let normalized = normalize_image(&image, config.normalization);
        let padded = reflect_pad_to_divisible(&normalized, 16);
        let (_, actual_h, actual_w) = padded.dim();
        if actual_h != padded_h || actual_w != padded_w {
            return Err(StarDistTrainError::Shape(
                "all batch entries must pad to the same spatial shape".to_string(),
            ));
        }
        for c in 0..channels {
            for y in 0..padded_h {
                for x in 0..padded_w {
                    input_batch[[batch_index, c, y, x]] = padded[[c, y, x]];
                }
            }
        }
    }

    let (raw, _) = input_batch.into_raw_vec_and_offset();
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(raw, [batch_size, channels, padded_h, padded_w]),
        device,
    );
    let (prob, dist) = model.forward(input);
    let prob_dims = prob.dims();
    let dist_dims = dist.dims();
    if prob_dims[0] != batch_size || prob_dims[1] != 1 {
        return Err(StarDistTrainError::Shape(format!(
            "prob output shape is incompatible with batch input: {:?}",
            prob_dims
        )));
    }
    let out_h = prob_dims[2];
    let out_w = prob_dims[3];
    if dist_dims[0] != batch_size
        || dist_dims[1] != config.n_rays
        || dist_dims[2] != out_h
        || dist_dims[3] != out_w
    {
        return Err(StarDistTrainError::Shape(format!(
            "prob/dist output shapes are incompatible: {:?} vs {:?}",
            prob_dims, dist_dims
        )));
    }

    let prob_vec = prob.into_data().into_vec().unwrap();
    let dist_vec = dist.into_data().into_vec().unwrap();
    let mut predictions = Vec::with_capacity(batch_size);
    for batch_index in 0..batch_size {
        let mut prob_arr = Array2::<f32>::zeros((out_h, out_w));
        let mut dist_arr = Array3::<f32>::zeros((out_h, out_w, config.n_rays));
        for y in 0..out_h {
            for x in 0..out_w {
                let prob_idx = (batch_index * out_h + y) * out_w + x;
                prob_arr[[y, x]] = prob_vec[prob_idx];
                for ray in 0..config.n_rays {
                    let dist_idx = (((batch_index * config.n_rays + ray) * out_h) + y) * out_w + x;
                    dist_arr[[y, x, ray]] = dist_vec[dist_idx];
                }
            }
        }
        predictions.push(Prediction2D {
            prob: prob_arr,
            dist: dist_arr,
        });
    }

    Ok(predictions)
}

/// Optimize probability and NMS thresholds on validation samples.
///
/// The score is the StarDist-style matching accuracy averaged over the supplied
/// IoU thresholds. `config.prob_threshold` and `config.nms_threshold` are
/// updated in place with the best pair.
pub fn optimize_thresholds_stardist_2d<B: Backend<FloatElem = f32, IntElem = i32>>(
    model: &TrainableStarDist2D<B>,
    samples: &[TrainingSample2D],
    config: &mut TrainingConfig2D,
    device: &B::Device,
    options: ThresholdOptimizationConfig2D,
) -> Result<PythonThresholds, StarDistTrainError> {
    config.validate()?;
    if samples.is_empty() {
        return Err(StarDistTrainError::EmptyDataset);
    }
    if options.prob_thresholds.is_empty()
        || options.nms_thresholds.is_empty()
        || options.iou_thresholds.is_empty()
    {
        return Err(StarDistTrainError::InvalidConfig(
            "threshold optimization candidate lists must not be empty".to_string(),
        ));
    }

    let max_samples = options
        .max_samples
        .unwrap_or(samples.len())
        .min(samples.len());
    let mut predictions = Vec::with_capacity(max_samples);
    for sample in samples.iter().take(max_samples) {
        let prediction = predict_stardist_2d_raw(model, &sample.image, config, device)?;
        predictions.push((prediction, sample));
    }

    let mut best_score = f32::NEG_INFINITY;
    let mut best_thresholds = PythonThresholds {
        prob: config.prob_threshold,
        nms: config.nms_threshold,
    };

    for &nms_threshold in &options.nms_thresholds {
        for &prob_threshold in &options.prob_thresholds {
            let mut score = 0.0;
            let mut count = 0usize;
            for (prediction, sample) in &predictions {
                let pred = labels_from_prob_dist_2d(
                    prediction.prob.clone(),
                    prediction.dist.clone(),
                    prob_threshold,
                    nms_threshold,
                    config.grid,
                    sample.labels.dim(),
                );
                let (gt, pred) = labels_for_matching(&sample.labels, &pred);
                for &iou_threshold in &options.iou_thresholds {
                    score += matching_accuracy(&gt, &pred, iou_threshold);
                    count += 1;
                }
            }

            let score = score / count.max(1) as f32;
            if score > best_score {
                best_score = score;
                best_thresholds = PythonThresholds {
                    prob: prob_threshold,
                    nms: nms_threshold,
                };
            }
        }
    }

    config.prob_threshold = best_thresholds.prob;
    config.nms_threshold = best_thresholds.nms;
    Ok(best_thresholds)
}

fn labels_for_matching(gt_signed: &Array2<i64>, pred: &Array2<u64>) -> (Array2<u64>, Array2<u64>) {
    let (height, width) = gt_signed.dim();
    let mut gt = Array2::<u64>::zeros((height, width));
    let mut pred_out = Array2::<u64>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let gt_value = gt_signed[[y, x]];
            if gt_value >= 0 {
                gt[[y, x]] = gt_value as u64;
                pred_out[[y, x]] = pred[[y, x]];
            }
        }
    }
    (gt, pred_out)
}

fn matching_accuracy(gt: &Array2<u64>, pred: &Array2<u64>, iou_threshold: f32) -> f32 {
    let mut gt_area = BTreeMap::<u64, usize>::new();
    let mut pred_area = BTreeMap::<u64, usize>::new();
    let mut intersections = BTreeMap::<(u64, u64), usize>::new();

    for ((y, x), &gt_id) in gt.indexed_iter() {
        let pred_id = pred[[y, x]];
        if gt_id > 0 {
            *gt_area.entry(gt_id).or_default() += 1;
        }
        if pred_id > 0 {
            *pred_area.entry(pred_id).or_default() += 1;
        }
        if gt_id > 0 && pred_id > 0 {
            *intersections.entry((gt_id, pred_id)).or_default() += 1;
        }
    }

    let mut candidates = Vec::new();
    for ((gt_id, pred_id), intersection) in intersections {
        let union = gt_area[&gt_id] + pred_area[&pred_id] - intersection;
        if union == 0 {
            continue;
        }
        let iou = intersection as f32 / union as f32;
        if iou >= iou_threshold {
            candidates.push((iou, gt_id, pred_id));
        }
    }
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut matched_gt = std::collections::BTreeSet::new();
    let mut matched_pred = std::collections::BTreeSet::new();
    for (_, gt_id, pred_id) in candidates {
        if matched_gt.contains(&gt_id) || matched_pred.contains(&pred_id) {
            continue;
        }
        matched_gt.insert(gt_id);
        matched_pred.insert(pred_id);
    }

    let true_positive = matched_gt.len();
    let false_negative = gt_area.len().saturating_sub(true_positive);
    let false_positive = pred_area.len().saturating_sub(matched_pred.len());
    let denom = true_positive + false_positive + false_negative;
    if denom == 0 {
        1.0
    } else {
        true_positive as f32 / denom as f32
    }
}

/// Save a trained StarDist2D model and config files.
///
/// Burn stores weights separately from the model definition. The config file is
/// therefore required when loading, because it reconstructs the same number of
/// channels, rays, grid, and inference thresholds before weights are applied.
/// This writes three files:
///
/// - `config.json`: Python StarDist `Config2D`-style metadata.
/// - `thresholds.json`: Python StarDist threshold metadata.
/// - `cellcast_config.json`: Rust-specific training/inference config.
///
/// The model file is saved with Burn's compact recorder. The recorder appends
/// its own extension, so pass an artifact directory, not a final file name.
pub fn save_stardist_2d<B: Backend, P: AsRef<Path>>(
    model: TrainableStarDist2D<B>,
    config: &TrainingConfig2D,
    artifact_dir: P,
) -> Result<(), StarDistTrainError> {
    let artifact_dir = artifact_dir.as_ref();
    fs::create_dir_all(artifact_dir)?;
    fs::write(
        artifact_dir.join("cellcast_config.json"),
        serde_json::to_string_pretty(config)?,
    )?;
    fs::write(
        artifact_dir.join("config.json"),
        serde_json::to_string_pretty(&config.to_python_config())?,
    )?;
    fs::write(
        artifact_dir.join("thresholds.json"),
        serde_json::to_string_pretty(&config.python_thresholds())?,
    )?;

    let recorder = CompactRecorder::new();
    model.save_file(artifact_dir.join("model"), &recorder)?;
    Ok(())
}

/// Load a saved StarDist2D model for the selected backend.
///
/// The same artifact directory can be loaded on CPU or WGPU as long as the
/// backend supports the device and the crate version still defines the same
/// model structure.
pub fn load_stardist_2d<B, P: AsRef<Path>>(
    artifact_dir: P,
    device: &B::Device,
) -> Result<(TrainableStarDist2D<B>, TrainingConfig2D), StarDistTrainError>
where
    B: Backend<FloatElem = f32, IntElem = i32>,
{
    let artifact_dir = artifact_dir.as_ref();
    let config: TrainingConfig2D =
        serde_json::from_slice(&fs::read(artifact_dir.join("cellcast_config.json"))?)?;
    config.validate()?;

    let model_config =
        TrainableStarDist2DConfig::new(config.n_channel_in, config.n_rays, config.grid);
    let model = model_config.init::<B>(device);
    let recorder = CompactRecorder::new();
    let model = model.load_file(artifact_dir.join("model"), &recorder, device)?;

    Ok((model, config))
}

fn validate_samples(
    samples: &[TrainingSample2D],
    config: &TrainingConfig2D,
) -> Result<(), StarDistTrainError> {
    if samples.is_empty() {
        return Err(StarDistTrainError::EmptyDataset);
    }
    for (index, sample) in samples.iter().enumerate() {
        let (channels, height, width) = sample.image.dim();
        let labels_shape = sample.labels.dim();
        if channels != config.n_channel_in {
            return Err(StarDistTrainError::Shape(format!(
                "sample {index} has {channels} channels, expected {}",
                config.n_channel_in
            )));
        }
        if labels_shape != (height, width) {
            return Err(StarDistTrainError::Shape(format!(
                "sample {index} image spatial shape ({height}, {width}) does not match labels {:?}",
                labels_shape
            )));
        }
        if height < config.patch_size[0] || width < config.patch_size[1] {
            return Err(StarDistTrainError::Shape(format!(
                "sample {index} shape ({height}, {width}) is smaller than patch_size {:?}",
                config.patch_size
            )));
        }
    }
    Ok(())
}

fn input_patch_size(config: &TrainingConfig2D) -> [usize; 2] {
    if config.shape_completion {
        [
            config.patch_size[0] - 2 * config.completion_crop,
            config.patch_size[1] - 2 * config.completion_crop,
        ]
    } else {
        config.patch_size
    }
}

fn build_batch_2d<B: Backend<FloatElem = f32, IntElem = i32>>(
    samples: &[TrainingSample2D],
    config: &TrainingConfig2D,
    device: &B::Device,
    rng: &mut StdRng,
    augment: bool,
) -> Result<Batch2D<B>, StarDistTrainError> {
    let batch = config.batch_size;
    let channels = config.n_channel_in;
    let [input_h, input_w] = input_patch_size(config);
    let out_h = input_h / config.grid[0];
    let out_w = input_w / config.grid[1];

    let mut images = Vec::with_capacity(batch * channels * input_h * input_w);
    let mut probs = Vec::with_capacity(batch * out_h * out_w);
    let mut prob_masks = Vec::with_capacity(batch * out_h * out_w);
    let mut dists = Vec::with_capacity(batch * config.n_rays * out_h * out_w);
    let mut masks = Vec::with_capacity(batch * out_h * out_w);

    for _ in 0..batch {
        let sample_index = rng.gen_range(0..samples.len());
        let sample = &samples[sample_index];
        let (mut image, mut labels) = crop_sample(sample, config, rng);
        if augment {
            augment_patch(&mut image, &mut labels, config.augment, rng)?;
        }
        if config.shape_completion {
            image = crop_image_center(&image, config.completion_crop);
        }
        image = normalize_image(&image, config.normalization);

        let target = build_target_2d(&labels, config)?;

        for c in 0..channels {
            for y in 0..input_h {
                for x in 0..input_w {
                    images.push(image[[c, y, x]]);
                }
            }
        }
        for y in 0..out_h {
            for x in 0..out_w {
                probs.push(target.prob[[0, y, x]]);
                prob_masks.push(target.prob_mask[[0, y, x]]);
            }
        }
        for ray in 0..config.n_rays {
            for y in 0..out_h {
                for x in 0..out_w {
                    dists.push(target.dist[[ray, y, x]]);
                }
            }
        }
        for y in 0..out_h {
            for x in 0..out_w {
                masks.push(target.dist_mask[[0, y, x]]);
            }
        }
    }

    Ok(Batch2D {
        image: Tensor::<B, 4>::from_data(
            TensorData::new(images, [batch, channels, input_h, input_w]),
            device,
        ),
        prob: Tensor::<B, 4>::from_data(TensorData::new(probs, [batch, 1, out_h, out_w]), device),
        prob_mask: Tensor::<B, 4>::from_data(
            TensorData::new(prob_masks, [batch, 1, out_h, out_w]),
            device,
        ),
        dist: Tensor::<B, 4>::from_data(
            TensorData::new(dists, [batch, config.n_rays, out_h, out_w]),
            device,
        ),
        dist_mask: Tensor::<B, 4>::from_data(
            TensorData::new(masks, [batch, 1, out_h, out_w]),
            device,
        ),
    })
}

fn stardist_loss<B: Backend>(
    prob_pred: Tensor<B, 4>,
    dist_pred: Tensor<B, 4>,
    prob_true: Tensor<B, 4>,
    prob_mask: Tensor<B, 4>,
    dist_true: Tensor<B, 4>,
    dist_mask: Tensor<B, 4>,
    config: &TrainingConfig2D,
) -> LossTensors<B> {
    let eps = 1e-7;
    let prob_pred = prob_pred.clamp(eps, 1.0 - eps);
    let one_prob = prob_true.clone().mul_scalar(0.0).add_scalar(1.0);
    let prob_pos = prob_true.clone() * prob_pred.clone().log();
    let prob_neg = (one_prob.clone() - prob_true) * (one_prob - prob_pred).log();
    let prob_bce = (prob_pos + prob_neg).neg();
    let prob_weight = prob_mask.clone().sum().clamp_min(eps);
    let prob_loss = (prob_bce * prob_mask).sum() / prob_weight;

    let rays = dist_pred.dims()[1];
    let mask = dist_mask.repeat(&[1, rays, 1, 1]);
    let one_mask = mask.clone().mul_scalar(0.0).add_scalar(1.0);
    let foreground_weight = mask.clone().sum().clamp_min(eps);
    let delta = dist_true - dist_pred.clone();
    let foreground_loss = (delta.abs() * mask.clone()).sum() / foreground_weight;
    let background_loss = (dist_pred.abs() * (one_mask - mask)).mean();
    let dist_loss = foreground_loss + background_loss.mul_scalar(config.background_reg);

    let total = prob_loss.clone().mul_scalar(config.loss_prob_weight)
        + dist_loss.clone().mul_scalar(config.loss_dist_weight);

    LossTensors {
        total,
        prob: prob_loss,
        dist: dist_loss,
    }
}

fn crop_sample(
    sample: &TrainingSample2D,
    config: &TrainingConfig2D,
    rng: &mut StdRng,
) -> (Array3<f32>, Array2<i64>) {
    let (_, height, width) = sample.image.dim();
    let patch_h = config.patch_size[0];
    let patch_w = config.patch_size[1];

    let foreground_only = rng.gen_bool(config.foreground_probability as f64);
    let (top, left) = sample_top_left_like_stardist(
        &sample.labels,
        height,
        width,
        patch_h,
        patch_w,
        foreground_only,
        rng,
    );

    let image = crop_image(&sample.image, top, left, patch_h, patch_w);
    let labels = crop_labels(&sample.labels, top, left, patch_h, patch_w);
    (image, labels)
}

fn sample_top_left_like_stardist(
    labels: &Array2<i64>,
    height: usize,
    width: usize,
    patch_h: usize,
    patch_w: usize,
    foreground_only: bool,
    rng: &mut StdRng,
) -> (usize, usize) {
    if foreground_only {
        let valid = valid_foreground_centers(labels, patch_h, patch_w);
        if let Some(&(y, x)) = valid.choose(rng) {
            return center_to_top_left(y, x, patch_h, patch_w);
        }
    }

    let y_min = patch_h / 2;
    let y_max = height - patch_h + patch_h / 2;
    let x_min = patch_w / 2;
    let x_max = width - patch_w + patch_w / 2;
    let center_y = rng.gen_range(y_min..=y_max);
    let center_x = rng.gen_range(x_min..=x_max);
    center_to_top_left(center_y, center_x, patch_h, patch_w)
}

fn valid_foreground_centers(
    labels: &Array2<i64>,
    patch_h: usize,
    patch_w: usize,
) -> Vec<(usize, usize)> {
    let (height, width) = labels.dim();
    let integral = positive_integral_image(labels);
    let y_min = patch_h / 2;
    let y_max = height - patch_h + patch_h / 2;
    let x_min = patch_w / 2;
    let x_max = width - patch_w + patch_w / 2;
    let mut centers = Vec::new();

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let (top, left) = center_to_top_left(y, x, patch_h, patch_w);
            if integral_sum(&integral, top, left, patch_h, patch_w) > 0 {
                centers.push((y, x));
            }
        }
    }

    centers
}

fn positive_integral_image(labels: &Array2<i64>) -> Array2<usize> {
    let (height, width) = labels.dim();
    let mut integral = Array2::<usize>::zeros((height + 1, width + 1));
    for y in 0..height {
        let mut row_sum = 0usize;
        for x in 0..width {
            if labels[[y, x]] > 0 {
                row_sum += 1;
            }
            integral[[y + 1, x + 1]] = integral[[y, x + 1]] + row_sum;
        }
    }
    integral
}

fn integral_sum(
    integral: &Array2<usize>,
    top: usize,
    left: usize,
    height: usize,
    width: usize,
) -> usize {
    let bottom = top + height;
    let right = left + width;
    integral[[bottom, right]] + integral[[top, left]]
        - integral[[top, right]]
        - integral[[bottom, left]]
}

fn center_to_top_left(y: usize, x: usize, patch_h: usize, patch_w: usize) -> (usize, usize) {
    (y - patch_h / 2, x - patch_w / 2)
}

fn crop_image(
    image: &Array3<f32>,
    top: usize,
    left: usize,
    patch_h: usize,
    patch_w: usize,
) -> Array3<f32> {
    let (channels, _, _) = image.dim();
    let mut out = Array3::<f32>::zeros((channels, patch_h, patch_w));
    for c in 0..channels {
        for y in 0..patch_h {
            for x in 0..patch_w {
                out[[c, y, x]] = image[[c, top + y, left + x]];
            }
        }
    }
    out
}

fn crop_image_center(image: &Array3<f32>, border: usize) -> Array3<f32> {
    let (_, height, width) = image.dim();
    crop_image(
        image,
        border,
        border,
        height - 2 * border,
        width - 2 * border,
    )
}

fn crop_labels(
    labels: &Array2<i64>,
    top: usize,
    left: usize,
    patch_h: usize,
    patch_w: usize,
) -> Array2<i64> {
    let mut out = Array2::<i64>::zeros((patch_h, patch_w));
    for y in 0..patch_h {
        for x in 0..patch_w {
            out[[y, x]] = labels[[top + y, left + x]];
        }
    }
    out
}

fn augment_patch(
    image: &mut Array3<f32>,
    labels: &mut Array2<i64>,
    augment: AugmentConfig2D,
    rng: &mut StdRng,
) -> Result<(), StarDistTrainError> {
    if augment.flip_y && rng.gen_bool(0.5) {
        flip_y(image, labels);
    }
    if augment.flip_x && rng.gen_bool(0.5) {
        flip_x(image, labels);
    }
    if augment.rotate90 && image.dim().1 == image.dim().2 {
        let turns = rng.gen_range(0..4);
        for _ in 0..turns {
            rotate90(image, labels);
        }
    }

    if augment.intensity_scale > 0.0 {
        let scale =
            rng.gen_range((1.0 - augment.intensity_scale)..=(1.0 + augment.intensity_scale));
        image.mapv_inplace(|v| v * scale);
    }
    if augment.intensity_shift > 0.0 {
        let shift = rng.gen_range(-augment.intensity_shift..=augment.intensity_shift);
        image.mapv_inplace(|v| v + shift);
    }
    if augment.gaussian_noise_std > 0.0 {
        image.mapv_inplace(|v| v + standard_normal(rng) * augment.gaussian_noise_std);
    }
    Ok(())
}

fn standard_normal(rng: &mut StdRng) -> f32 {
    let u1 = rng.gen_range(f32::MIN_POSITIVE..1.0);
    let u2 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

fn flip_y(image: &mut Array3<f32>, labels: &mut Array2<i64>) {
    let (channels, height, width) = image.dim();
    let image_copy = image.clone();
    let labels_copy = labels.clone();
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                image[[c, y, x]] = image_copy[[c, height - 1 - y, x]];
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            labels[[y, x]] = labels_copy[[height - 1 - y, x]];
        }
    }
}

fn flip_x(image: &mut Array3<f32>, labels: &mut Array2<i64>) {
    let (channels, height, width) = image.dim();
    let image_copy = image.clone();
    let labels_copy = labels.clone();
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                image[[c, y, x]] = image_copy[[c, y, width - 1 - x]];
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            labels[[y, x]] = labels_copy[[y, width - 1 - x]];
        }
    }
}

fn rotate90(image: &mut Array3<f32>, labels: &mut Array2<i64>) {
    let (channels, height, width) = image.dim();
    let image_copy = image.clone();
    let labels_copy = labels.clone();
    for c in 0..channels {
        for y in 0..height {
            for x in 0..width {
                image[[c, y, x]] = image_copy[[c, width - 1 - x, y]];
            }
        }
    }
    for y in 0..height {
        for x in 0..width {
            labels[[y, x]] = labels_copy[[width - 1 - x, y]];
        }
    }
}

struct Target2D {
    prob: Array3<f32>,
    prob_mask: Array3<f32>,
    dist: Array3<f32>,
    dist_mask: Array3<f32>,
}

fn build_target_2d(
    labels: &Array2<i64>,
    config: &TrainingConfig2D,
) -> Result<Target2D, StarDistTrainError> {
    let grid = config.grid;
    let border = if config.shape_completion {
        config.completion_crop
    } else {
        0
    };
    let (height, width) = labels.dim();
    let target_h = height - 2 * border;
    let target_w = width - 2 * border;
    let clean = positive_labels(labels);
    let central_clean = crop_labels_u64(&clean, border, border, target_h, target_w);
    let prob_labels = downsample_labels_u64(&central_clean, grid);
    let prob_grid = edt_prob_2d(&prob_labels);
    let out_h = prob_grid.dim().0;
    let out_w = prob_grid.dim().1;
    let mut prob = Array3::<f32>::zeros((1, out_h, out_w));
    let mut prob_mask = Array3::<f32>::ones((1, out_h, out_w));
    let mut dist_mask = Array3::<f32>::zeros((1, out_h, out_w));
    let dist_mask_grid = if config.shape_completion {
        let cleared = clear_border_labels_u64(&clean);
        let central_cleared = crop_labels_u64(&cleared, border, border, target_h, target_w);
        edt_prob_2d(&downsample_labels_u64(&central_cleared, grid))
    } else {
        prob_grid.clone()
    };

    for y in 0..out_h {
        for x in 0..out_w {
            let p = prob_grid[[y, x]];
            prob[[0, y, x]] = p;
            dist_mask[[0, y, x]] = dist_mask_grid[[y, x]];
            let yy = border + y * grid[0];
            let xx = border + x * grid[1];
            if labels[[yy, xx]] < 0 {
                prob_mask[[0, y, x]] = 0.0;
            }
        }
    }

    let dist = if config.shape_completion {
        let cleared = clear_border_labels_u64(&clean);
        let full_dist = star_dist_2d(&cleared, config.n_rays, [1, 1])?;
        subsample_dist_center(&full_dist, border, target_h, target_w, grid)
    } else {
        star_dist_2d(&clean, config.n_rays, grid)?
    };

    Ok(Target2D {
        prob,
        prob_mask,
        dist,
        dist_mask,
    })
}

fn positive_labels(labels: &Array2<i64>) -> Array2<u64> {
    labels.mapv(|label| if label > 0 { label as u64 } else { 0 })
}

fn downsample_labels_u64(labels: &Array2<u64>, grid: [usize; 2]) -> Array2<u64> {
    let (height, width) = labels.dim();
    let out_h = height / grid[0];
    let out_w = width / grid[1];
    let mut out = Array2::<u64>::zeros((out_h, out_w));
    for y in 0..out_h {
        for x in 0..out_w {
            out[[y, x]] = labels[[y * grid[0], x * grid[1]]];
        }
    }
    out
}

fn crop_labels_u64(
    labels: &Array2<u64>,
    top: usize,
    left: usize,
    height: usize,
    width: usize,
) -> Array2<u64> {
    let mut out = Array2::<u64>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            out[[y, x]] = labels[[top + y, left + x]];
        }
    }
    out
}

fn clear_border_labels_u64(labels: &Array2<u64>) -> Array2<u64> {
    let (height, width) = labels.dim();
    let mut border_labels = std::collections::BTreeSet::new();
    for x in 0..width {
        if labels[[0, x]] > 0 {
            border_labels.insert(labels[[0, x]]);
        }
        if labels[[height - 1, x]] > 0 {
            border_labels.insert(labels[[height - 1, x]]);
        }
    }
    for y in 0..height {
        if labels[[y, 0]] > 0 {
            border_labels.insert(labels[[y, 0]]);
        }
        if labels[[y, width - 1]] > 0 {
            border_labels.insert(labels[[y, width - 1]]);
        }
    }

    let mut out = labels.clone();
    for value in out.iter_mut() {
        if border_labels.contains(value) {
            *value = 0;
        }
    }
    out
}

fn subsample_dist_center(
    dist: &Array3<f32>,
    border: usize,
    height: usize,
    width: usize,
    grid: [usize; 2],
) -> Array3<f32> {
    let (rays, _, _) = dist.dim();
    let out_h = height / grid[0];
    let out_w = width / grid[1];
    let mut out = Array3::<f32>::zeros((rays, out_h, out_w));
    for ray in 0..rays {
        for y in 0..out_h {
            for x in 0..out_w {
                out[[ray, y, x]] = dist[[ray, border + y * grid[0], border + x * grid[1]]];
            }
        }
    }
    out
}

fn edt_prob_2d(labels: &Array2<u64>) -> Array2<f32> {
    let (height, width) = labels.dim();
    let mut prob = Array2::<f32>::zeros((height, width));
    let boxes = object_boxes(labels);

    for (label, bbox) in boxes {
        let local_h = bbox.height() + 2;
        let local_w = bbox.width() + 2;
        let mut foreground = vec![false; local_h * local_w];
        for y in bbox.y_min..=bbox.y_max {
            for x in bbox.x_min..=bbox.x_max {
                if labels[[y, x]] == label {
                    let local_y = y - bbox.y_min + 1;
                    let local_x = x - bbox.x_min + 1;
                    foreground[local_y * local_w + local_x] = true;
                }
            }
        }

        let dist_sq = distance_to_background_squared(&foreground, local_h, local_w);
        let mut max_distance = 0.0_f32;
        for y in bbox.y_min..=bbox.y_max {
            for x in bbox.x_min..=bbox.x_max {
                if labels[[y, x]] == label {
                    let local_y = y - bbox.y_min + 1;
                    let local_x = x - bbox.x_min + 1;
                    let distance = dist_sq[local_y * local_w + local_x].sqrt();
                    max_distance = max_distance.max(distance);
                }
            }
        }

        if max_distance > 0.0 {
            for y in bbox.y_min..=bbox.y_max {
                for x in bbox.x_min..=bbox.x_max {
                    if labels[[y, x]] == label {
                        let local_y = y - bbox.y_min + 1;
                        let local_x = x - bbox.x_min + 1;
                        let distance = dist_sq[local_y * local_w + local_x].sqrt();
                        prob[[y, x]] = distance / max_distance;
                    }
                }
            }
        }
    }

    prob
}

#[derive(Clone, Copy, Debug)]
struct ObjectBox {
    y_min: usize,
    y_max: usize,
    x_min: usize,
    x_max: usize,
}

impl ObjectBox {
    fn new(y: usize, x: usize) -> Self {
        Self {
            y_min: y,
            y_max: y,
            x_min: x,
            x_max: x,
        }
    }

    fn include(&mut self, y: usize, x: usize) {
        self.y_min = self.y_min.min(y);
        self.y_max = self.y_max.max(y);
        self.x_min = self.x_min.min(x);
        self.x_max = self.x_max.max(x);
    }

    fn height(&self) -> usize {
        self.y_max - self.y_min + 1
    }

    fn width(&self) -> usize {
        self.x_max - self.x_min + 1
    }
}

fn object_boxes(labels: &Array2<u64>) -> BTreeMap<u64, ObjectBox> {
    let mut boxes = BTreeMap::new();
    for ((y, x), &label) in labels.indexed_iter() {
        if label == 0 {
            continue;
        }
        boxes
            .entry(label)
            .and_modify(|bbox: &mut ObjectBox| bbox.include(y, x))
            .or_insert_with(|| ObjectBox::new(y, x));
    }
    boxes
}

fn distance_to_background_squared(foreground: &[bool], height: usize, width: usize) -> Vec<f32> {
    const INF: f32 = 1.0e20;
    let mut column_input = vec![0.0; height];
    let mut column_output = vec![0.0; height];
    let mut tmp = vec![0.0; height * width];

    for x in 0..width {
        for y in 0..height {
            column_input[y] = if foreground[y * width + x] { INF } else { 0.0 };
        }
        edt_1d_squared(&column_input, &mut column_output);
        for y in 0..height {
            tmp[y * width + x] = column_output[y];
        }
    }

    let mut row_output = vec![0.0; width];
    let mut out = vec![0.0; height * width];
    for y in 0..height {
        let row_input = &tmp[y * width..(y + 1) * width];
        edt_1d_squared(row_input, &mut row_output);
        out[y * width..(y + 1) * width].copy_from_slice(&row_output);
    }

    out
}

fn edt_1d_squared(input: &[f32], output: &mut [f32]) {
    let n = input.len();
    if n == 0 {
        return;
    }

    let mut locations = vec![0usize; n];
    let mut boundaries = vec![0.0_f32; n + 1];
    let mut segment = 0usize;
    locations[0] = 0;
    boundaries[0] = f32::NEG_INFINITY;
    boundaries[1] = f32::INFINITY;

    for q in 1..n {
        let mut intersection;
        loop {
            let p = locations[segment];
            intersection = parabola_intersection(input, q, p);
            if intersection > boundaries[segment] {
                break;
            }
            if segment == 0 {
                break;
            }
            segment -= 1;
        }
        if intersection <= boundaries[segment] {
            segment = 0;
            intersection = parabola_intersection(input, q, locations[segment]);
        }
        segment += 1;
        locations[segment] = q;
        boundaries[segment] = intersection;
        boundaries[segment + 1] = f32::INFINITY;
    }

    segment = 0;
    for (q, value) in output.iter_mut().enumerate() {
        while boundaries[segment + 1] < q as f32 {
            segment += 1;
        }
        let p = locations[segment];
        let delta = q as f32 - p as f32;
        *value = delta * delta + input[p];
    }
}

fn parabola_intersection(input: &[f32], q: usize, p: usize) -> f32 {
    let qf = q as f32;
    let pf = p as f32;
    ((input[q] + qf * qf) - (input[p] + pf * pf)) / (2.0 * (qf - pf))
}

fn star_dist_2d(
    labels: &Array2<u64>,
    n_rays: usize,
    grid: [usize; 2],
) -> Result<Array3<f32>, StarDistTrainError> {
    let (height, width) = labels.dim();
    if height % grid[0] != 0 || width % grid[1] != 0 {
        return Err(StarDistTrainError::Shape(format!(
            "label shape ({height}, {width}) must be divisible by grid {:?}",
            grid
        )));
    }
    let out_h = height / grid[0];
    let out_w = width / grid[1];
    let phis: Vec<f32> = (0..n_rays)
        .map(|ray| 2.0 * std::f32::consts::PI * ray as f32 / n_rays as f32)
        .collect();
    let labels_flat: Vec<u64> = labels.iter().copied().collect();
    let output_point_count = out_h * out_w;
    let mut point_major = vec![0.0_f32; output_point_count * n_rays];

    point_major
        .par_chunks_mut(n_rays)
        .enumerate()
        .for_each(|(point_index, distances)| {
            let y = point_index / out_w;
            let x = point_index % out_w;
            let y0 = y * grid[0];
            let x0 = x * grid[1];
            let label = labels_flat[y0 * width + x0];
            if label == 0 {
                return;
            }

            for (ray, phi) in phis.iter().enumerate() {
                let dy = phi.sin();
                let dx = phi.cos();
                let mut yy_rel = 0.0_f32;
                let mut xx_rel = 0.0_f32;
                loop {
                    yy_rel += dy;
                    xx_rel += dx;
                    let yy = y0 as isize + yy_rel.round() as isize;
                    let xx = x0 as isize + xx_rel.round() as isize;
                    if yy < 0
                        || xx < 0
                        || yy >= height as isize
                        || xx >= width as isize
                        || labels_flat[yy as usize * width + xx as usize] != label
                    {
                        let correction = 1.0 - 0.5 / dy.abs().max(dx.abs());
                        yy_rel -= correction * dy;
                        xx_rel -= correction * dx;
                        distances[ray] = (yy_rel * yy_rel + xx_rel * xx_rel).sqrt();
                        break;
                    }
                }
            }
        });

    let mut out = Array3::<f32>::zeros((n_rays, out_h, out_w));

    for y in 0..out_h {
        for x in 0..out_w {
            let point_index = y * out_w + x;
            for ray in 0..n_rays {
                out[[ray, y, x]] = point_major[point_index * n_rays + ray];
            }
        }
    }

    Ok(out)
}

fn normalize_image(image: &Array3<f32>, normalization: Normalization2D) -> Array3<f32> {
    match normalization {
        Normalization2D::None => image.clone(),
        Normalization2D::Percentile { pmin, pmax } => {
            percentile_normalize_channel_first(image, pmin, pmax)
        }
    }
}

fn percentile_normalize_channel_first(image: &Array3<f32>, pmin: f32, pmax: f32) -> Array3<f32> {
    let (channels, height, width) = image.dim();
    let mut out = Array3::<f32>::zeros((channels, height, width));
    for c in 0..channels {
        let mut values = Vec::with_capacity(height * width);
        for y in 0..height {
            for x in 0..width {
                let value = image[[c, y, x]];
                if value.is_finite() {
                    values.push(value);
                }
            }
        }
        if values.is_empty() {
            continue;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo = percentile_from_sorted(&values, pmin);
        let hi = percentile_from_sorted(&values, pmax);
        let scale = (hi - lo).max(1e-6);
        for y in 0..height {
            for x in 0..width {
                out[[c, y, x]] = ((image[[c, y, x]] - lo) / scale).clamp(0.0, 1.0);
            }
        }
    }
    out
}

fn percentile_from_sorted(values: &[f32], percentile: f32) -> f32 {
    let percentile = percentile.clamp(0.0, 100.0);
    let pos = (percentile / 100.0) * (values.len().saturating_sub(1) as f32);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        let weight = pos - lo as f32;
        values[lo] * (1.0 - weight) + values[hi] * weight
    }
}

fn reflect_pad_to_divisible(image: &Array3<f32>, div_by: usize) -> Array3<f32> {
    let (channels, height, width) = image.dim();
    let pad_h = (div_by - (height % div_by)) % div_by;
    let pad_w = (div_by - (width % div_by)) % div_by;
    if pad_h == 0 && pad_w == 0 {
        return image.clone();
    }
    let out_h = height + pad_h;
    let out_w = width + pad_w;
    let mut out = Array3::<f32>::zeros((channels, out_h, out_w));
    for c in 0..channels {
        for y in 0..out_h {
            let yy = reflect_index(y, height);
            for x in 0..out_w {
                let xx = reflect_index(x, width);
                out[[c, y, x]] = image[[c, yy, xx]];
            }
        }
    }
    out
}

fn reflect_index(index: usize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * len - 2;
    let index = index % period;
    if index < len { index } else { period - index }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_generation_is_channel_first() {
        let mut labels = Array2::<i64>::zeros((32, 32));
        for y in 8..24 {
            for x in 8..24 {
                labels[[y, x]] = 1;
            }
        }
        let config = TrainingConfig2D {
            n_rays: 8,
            grid: [2, 2],
            patch_size: [32, 32],
            ..Default::default()
        };

        let target = build_target_2d(&labels, &config).unwrap();

        assert_eq!(target.prob.dim(), (1, 16, 16));
        assert_eq!(target.prob_mask.dim(), (1, 16, 16));
        assert_eq!(target.dist.dim(), (8, 16, 16));
        assert_eq!(target.dist_mask.dim(), (1, 16, 16));
        assert!(target.prob[[0, 8, 8]] > target.prob[[0, 4, 4]]);
        assert!(target.dist[[0, 8, 8]] > 0.0);
    }

    #[test]
    fn reflect_padding_keeps_channel_first_shape() {
        let image = Array3::<f32>::from_shape_vec(
            (1, 3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();

        let padded = reflect_pad_to_divisible(&image, 4);

        assert_eq!(padded.dim(), (1, 4, 4));
        assert_eq!(padded[[0, 0, 0]], 1.0);
        assert_eq!(padded[[0, 3, 3]], 5.0);
    }

    #[test]
    fn sample_validation_rejects_channel_mismatch() {
        let image = Array3::<f32>::zeros((2, 32, 32));
        let labels = Array2::<i64>::zeros((32, 32));
        let sample = TrainingSample2D::new(image, labels).unwrap();
        let config = TrainingConfig2D {
            n_channel_in: 1,
            patch_size: [32, 32],
            ..Default::default()
        };

        assert!(validate_samples(&[sample], &config).is_err());
    }
}
