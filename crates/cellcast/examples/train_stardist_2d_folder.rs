use std::env;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::path::PathBuf;

use cellcast::training::io_2d::{
    load_training_dataset_from_folder_with_options,
    load_training_samples_from_folders_with_options, split_train_valid_2d, FolderDatasetOptions2D,
    ImageChannels2D, LabelColorMode2D,
};
use cellcast::training::stardist_2d::{
    save_stardist_2d, train_stardist_2d_cpu, train_stardist_2d_wgpu, LearningRateSchedule2D,
    Normalization2D, TrainingConfig2D,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse().map_err(|msg| IoError::new(ErrorKind::InvalidInput, msg))?;
    let options = FolderDatasetOptions2D {
        image_channels: args.image_channels,
        label_color_mode: args.label_color_mode,
    };

    let (train, valid) = if let Some(gt_dir) = args.gt_dir.as_ref() {
        let samples =
            load_training_samples_from_folders_with_options(&args.data_dir, gt_dir, options)?;
        split_train_valid_2d(samples, args.validation_fraction, args.seed)?
    } else {
        let split = load_training_dataset_from_folder_with_options(
            &args.data_dir,
            options,
            args.validation_fraction,
            args.seed,
        )?;
        (split.train_samples, split.valid_samples)
    };
    let channels = train
        .first()
        .or_else(|| valid.first())
        .map(|sample| sample.image.dim().0)
        .ok_or_else(|| IoError::new(ErrorKind::InvalidInput, "dataset is empty"))?;

    let config = TrainingConfig2D {
        n_channel_in: channels,
        n_rays: args.n_rays,
        grid: [args.grid, args.grid],
        patch_size: [args.patch_size, args.patch_size],
        batch_size: args.batch_size,
        epochs: args.epochs,
        steps_per_epoch: args.steps_per_epoch,
        validation_steps: args.validation_steps,
        lr_schedule: args.lr_schedule,
        shape_completion: args.shape_completion,
        completion_crop: args.completion_crop,
        normalization: args.normalization,
        seed: args.seed,
        ..Default::default()
    };
    let valid_ref = if valid.is_empty() {
        None
    } else {
        Some(valid.as_slice())
    };

    println!(
        "training StarDist2D on {} train / {} validation images with {} channel(s)",
        train.len(),
        valid.len(),
        channels
    );

    if args.cpu {
        let result = train_stardist_2d_cpu(&train, valid_ref, config)?;
        save_stardist_2d(result.model, &result.config, &args.output_dir)?;
    } else {
        let result = train_stardist_2d_wgpu(&train, valid_ref, config)?;
        save_stardist_2d(result.model, &result.config, &args.output_dir)?;
    }

    println!("saved model artifacts to {}", args.output_dir.display());
    Ok(())
}

#[derive(Debug)]
struct Args {
    data_dir: PathBuf,
    gt_dir: Option<PathBuf>,
    output_dir: PathBuf,
    cpu: bool,
    image_channels: ImageChannels2D,
    label_color_mode: LabelColorMode2D,
    epochs: usize,
    steps_per_epoch: usize,
    validation_steps: usize,
    batch_size: usize,
    patch_size: usize,
    grid: usize,
    n_rays: usize,
    validation_fraction: f32,
    lr_schedule: LearningRateSchedule2D,
    poly_power: f64,
    normalization: Normalization2D,
    shape_completion: bool,
    completion_crop: usize,
    seed: u64,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut raw = env::args().skip(1);
        let mut positionals = Vec::new();
        let mut args = Args {
            data_dir: PathBuf::new(),
            gt_dir: None,
            output_dir: PathBuf::new(),
            cpu: false,
            image_channels: ImageChannels2D::Auto,
            label_color_mode: LabelColorMode2D::Auto,
            epochs: 100,
            steps_per_epoch: 100,
            validation_steps: 10,
            batch_size: 4,
            patch_size: 256,
            grid: 1,
            n_rays: 32,
            validation_fraction: 0.15,
            lr_schedule: LearningRateSchedule2D::default(),
            poly_power: 0.9,
            normalization: Normalization2D::None,
            shape_completion: false,
            completion_crop: 32,
            seed: 42,
        };

        while let Some(arg) = raw.next() {
            match arg.as_str() {
                "--cpu" => args.cpu = true,
                "--gray" | "--grayscale" => args.image_channels = ImageChannels2D::Grayscale,
                "--rgb" => args.image_channels = ImageChannels2D::Rgb,
                "--label-color-ids" => args.label_color_mode = LabelColorMode2D::ColorIds,
                "--label-grayscale" => args.label_color_mode = LabelColorMode2D::Grayscale,
                "--normalize-percentile" => {
                    args.normalization = Normalization2D::Percentile {
                        pmin: 1.0,
                        pmax: 99.8,
                    };
                }
                "--lr-constant" => args.lr_schedule = LearningRateSchedule2D::Constant,
                "--lr-cosine" => {
                    let min_learning_rate = parse_next(&mut raw, "--lr-cosine")?;
                    args.lr_schedule =
                        LearningRateSchedule2D::CosineAnnealing { min_learning_rate };
                }
                "--lr-poly" => {
                    let min_learning_rate = parse_next(&mut raw, "--lr-poly")?;
                    args.lr_schedule = LearningRateSchedule2D::PolynomialDecay {
                        power: args.poly_power,
                        min_learning_rate,
                    };
                }
                "--poly-power" => args.poly_power = parse_next(&mut raw, "--poly-power")?,
                "--shape-completion" => args.shape_completion = true,
                "--epochs" => args.epochs = parse_next(&mut raw, "--epochs")?,
                "--steps" => args.steps_per_epoch = parse_next(&mut raw, "--steps")?,
                "--valid-steps" => args.validation_steps = parse_next(&mut raw, "--valid-steps")?,
                "--batch" => args.batch_size = parse_next(&mut raw, "--batch")?,
                "--patch" => args.patch_size = parse_next(&mut raw, "--patch")?,
                "--grid" => args.grid = parse_next(&mut raw, "--grid")?,
                "--rays" => args.n_rays = parse_next(&mut raw, "--rays")?,
                "--valid-frac" | "--val-frac" | "--validation-fraction" => {
                    let value = parse_next(&mut raw, arg.as_str())?;
                    args.validation_fraction = parse_validation_fraction(value)?;
                }
                "--valid-percent"
                | "--val-percent"
                | "--validation-percent"
                | "--valid-percentage"
                | "--val-percentage"
                | "--validation-percentage" => {
                    let value = parse_next(&mut raw, arg.as_str())?;
                    args.validation_fraction = parse_validation_percent(value)?;
                }
                "--completion-crop" => {
                    args.completion_crop = parse_next(&mut raw, "--completion-crop")?
                }
                "--seed" => args.seed = parse_next(&mut raw, "--seed")?,
                "--help" | "-h" => return Err(usage()),
                _ if arg.starts_with('-') => {
                    return Err(format!("unknown argument '{arg}'\n\n{}", usage()));
                }
                _ => positionals.push(PathBuf::from(arg)),
            }
        }

        match positionals.as_slice() {
            [dataset_dir, output_dir] => {
                args.data_dir = dataset_dir.clone();
                args.output_dir = output_dir.clone();
            }
            [data_dir, gt_dir, output_dir] => {
                args.data_dir = data_dir.clone();
                args.gt_dir = Some(gt_dir.clone());
                args.output_dir = output_dir.clone();
            }
            _ => return Err(usage()),
        }

        if let LearningRateSchedule2D::PolynomialDecay {
            min_learning_rate, ..
        } = args.lr_schedule
        {
            args.lr_schedule = LearningRateSchedule2D::PolynomialDecay {
                power: args.poly_power,
                min_learning_rate,
            };
        }

        if args.grid != 1 && args.grid != 2 {
            return Err("--grid must be 1 or 2".to_string());
        }
        args.validation_fraction = parse_validation_fraction(args.validation_fraction)?;
        if !args.shape_completion && args.patch_size % 16 != 0 {
            return Err("--patch must be divisible by 16".to_string());
        }
        if args.shape_completion {
            let input_patch = args
                .patch_size
                .checked_sub(2 * args.completion_crop)
                .ok_or_else(|| "--patch must be larger than 2 * --completion-crop".to_string())?;
            if input_patch % 16 != 0 {
                return Err("--patch - 2 * --completion-crop must be divisible by 16".to_string());
            }
        }

        Ok(args)
    }
}

fn parse_validation_fraction(value: f32) -> Result<f32, String> {
    if (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err("validation fraction must be in [0, 1)".to_string())
    }
}

fn parse_validation_percent(value: f32) -> Result<f32, String> {
    if (0.0..100.0).contains(&value) {
        Ok(value / 100.0)
    } else {
        Err("validation percentage must be in [0, 100)".to_string())
    }
}

fn parse_next<T>(raw: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = raw
        .next()
        .ok_or_else(|| format!("missing value after {flag}\n\n{}", usage()))?;
    value
        .parse::<T>()
        .map_err(|err| format!("invalid value for {flag}: {err}"))
}

fn usage() -> String {
    "Usage:
  cargo run --release -p cellcast --example train_stardist_2d_folder -- <dataset_dir> <output_dir> [options]
  cargo run --release -p cellcast --example train_stardist_2d_folder -- <data_dir> <gt_dir> <output_dir> [options]

Required:
  <dataset_dir>   Dataset root. May contain train/ and val|valid|validation/
                  splits, same-folder image/mask files, or data/gt-style
                  subfolders.
  <data_dir>      Explicit folder with input images, for example dataset/data
  <gt_dir>        Explicit folder with instance masks, for example dataset/gt
  <output_dir>    Folder where model artifacts will be written

Options:
  --cpu                 Train on CPU instead of WebGPU
  --gray                Force input images to one grayscale channel
  --rgb                 Force input images to three RGB channels
  --label-color-ids     Treat colored GT masks as color-coded instance ids
  --label-grayscale     Require GT masks to be grayscale labels
  --normalize-percentile
                        Apply 1/99.8 percentile normalization in Rust
  --lr-constant         Disable LR scheduling
  --lr-cosine MIN_LR    Use cosine annealing down to MIN_LR
  --lr-poly MIN_LR      Use polynomial decay down to MIN_LR
  --poly-power P        Polynomial decay power. Default: 0.9
  --shape-completion    Train complete shapes from center crops
  --epochs N            Default: 100
  --steps N             Steps per epoch. Default: 100
  --valid-steps N       Validation steps per epoch. Default: 10
  --batch N             Default: 4
  --patch N             Square patch size, divisible by 16. Default: 256
  --grid 1|2            StarDist prediction grid. Default: 1
  --rays N              Number of 2D rays. Default: 32
  --valid-frac F        Random validation fraction in [0, 1). Default: 0.15
  --valid-percent P     Random validation percentage in [0, 100)
  --completion-crop N   Shape-completion crop. Default: 32
  --seed N              Default: 42"
        .to_string()
}
