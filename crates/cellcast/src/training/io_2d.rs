//! Image-folder readers for 2D StarDist training.
//!
//! The trainer itself works with in-memory `TrainingSample2D` values. This
//! module is the thin IO layer that turns common image files into that training
//! representation:
//!
//! - input images become channel-first `Array3<f32>` with shape
//!   `[channel, y, x]`;
//! - ground-truth instance masks become `Array2<i64>` with shape `[y, x]`;
//! - paired folders are matched by file stem, independent of extension.
//!
//! The expected folder layout is:
//!
//! ```text
//! dataset/
//!   data/
//!     img_001.tif
//!     img_002.png
//!   gt/
//!     img_001.tif
//!     img_002.png
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use image::{DynamicImage, ImageBuffer, ImageReader, Luma, LumaA, Rgb, Rgba};
use ndarray::{Array2, Array3};
use rand::prelude::*;
use rand::seq::SliceRandom;

use crate::training::stardist_2d::{StarDistTrainError, TrainingSample2D};

/// File extensions accepted by the folder loader.
///
/// Format decoding is provided by the `image` crate. TIFF, PNG, and JPEG cover
/// the common microscopy and quick-preview paths; BMP and PNM are included
/// because they are simple uncompressed interchange formats.
pub const SUPPORTED_IMAGE_EXTENSIONS_2D: &[&str] = &[
    "tif", "tiff", "png", "jpg", "jpeg", "bmp", "pbm", "pgm", "ppm", "pnm",
];

/// How input image channels should be loaded.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ImageChannels2D {
    /// Preserve grayscale images as one channel and RGB/RGBA images as three
    /// channels. Alpha channels are ignored.
    #[default]
    Auto,
    /// Convert every image to one grayscale channel.
    Grayscale,
    /// Convert every image to three RGB channels. Alpha channels are ignored.
    Rgb,
}

/// How color ground-truth masks should be interpreted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LabelColorMode2D {
    /// Use a single channel when all RGB channels are equal; otherwise encode
    /// every unique non-black color as a distinct instance id.
    #[default]
    Auto,
    /// Require RGB masks to be grayscale-like and use the first channel as the
    /// instance id.
    Grayscale,
    /// Encode RGB colors as integer instance ids. Black remains background 0.
    ColorIds,
}

/// Options used when reading a paired image/mask folder dataset.
#[derive(Clone, Debug, Default)]
pub struct FolderDatasetOptions2D {
    pub image_channels: ImageChannels2D,
    pub label_color_mode: LabelColorMode2D,
}

/// One image/mask pair found in paired data and ground-truth folders.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainingFilePair2D {
    pub stem: String,
    pub image_path: PathBuf,
    pub label_path: PathBuf,
}

/// Read paired image and ground-truth folders using default options.
pub fn load_training_samples_from_folders<D, G>(
    data_dir: D,
    gt_dir: G,
) -> Result<Vec<TrainingSample2D>, StarDistTrainError>
where
    D: AsRef<Path>,
    G: AsRef<Path>,
{
    load_training_samples_from_folders_with_options(
        data_dir,
        gt_dir,
        FolderDatasetOptions2D::default(),
    )
}

/// Read paired image and ground-truth folders using explicit channel options.
pub fn load_training_samples_from_folders_with_options<D, G>(
    data_dir: D,
    gt_dir: G,
    options: FolderDatasetOptions2D,
) -> Result<Vec<TrainingSample2D>, StarDistTrainError>
where
    D: AsRef<Path>,
    G: AsRef<Path>,
{
    let pairs = collect_training_file_pairs_2d(data_dir, gt_dir)?;
    let mut samples = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let image = read_image_2d(&pair.image_path, options.image_channels)?;
        let labels = read_label_image_2d(&pair.label_path, options.label_color_mode)?;
        let sample = TrainingSample2D::new(image, labels).map_err(|err| {
            StarDistTrainError::Dataset(format!(
                "invalid pair '{}' (image {}, labels {}): {err}",
                pair.stem,
                pair.image_path.display(),
                pair.label_path.display()
            ))
        })?;
        samples.push(sample);
    }

    Ok(samples)
}

/// Pair images and labels by file stem.
///
/// Extensions do not need to match. For example, `data/img_001.tif` pairs with
/// `gt/img_001.png`.
pub fn collect_training_file_pairs_2d<D, G>(
    data_dir: D,
    gt_dir: G,
) -> Result<Vec<TrainingFilePair2D>, StarDistTrainError>
where
    D: AsRef<Path>,
    G: AsRef<Path>,
{
    let data_dir = data_dir.as_ref();
    let gt_dir = gt_dir.as_ref();
    let images = collect_supported_files(data_dir, "data")?;
    let mut labels = collect_supported_files(gt_dir, "gt")?;
    let mut pairs = Vec::with_capacity(images.len());

    for (stem, image_path) in images {
        let label_path = labels.remove(&stem).ok_or_else(|| {
            StarDistTrainError::Dataset(format!(
                "no ground-truth mask found for '{}' from {}",
                stem,
                image_path.display()
            ))
        })?;
        pairs.push(TrainingFilePair2D {
            stem,
            image_path,
            label_path,
        });
    }

    if !labels.is_empty() {
        let unmatched = labels
            .keys()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        return Err(StarDistTrainError::Dataset(format!(
            "found {} ground-truth masks without matching data images; examples: {}",
            labels.len(),
            unmatched
        )));
    }

    Ok(pairs)
}

/// Read one 2D image as a channel-first `f32` ndarray.
///
/// This is still plain Rust data loading code, not Burn code yet. Burn enters
/// later, when the trainer converts this `Array3<f32>` into a Burn tensor.
pub fn read_image_2d<P: AsRef<Path>>(
    // `path` is generic so callers can pass `&str`, `String`, `PathBuf`, or
    // `&Path`. The `AsRef<Path>` bound above guarantees that all of those can
    // be viewed as a filesystem path without forcing one exact input type.
    path: P,
    // `channels` tells the loader whether to preserve the source channel count
    // automatically, force one grayscale channel, or force three RGB channels.
    channels: ImageChannels2D,
    // Returning `Result` is the standard Rust way to represent an operation
    // that can fail. On success we return `Array3<f32>` with shape
    // `[channel, y, x]`; on failure we return our training error type.
) -> Result<Array3<f32>, StarDistTrainError> {
    // Convert the generic path-like value into `&Path`, which is the concrete
    // path reference expected by the lower-level image reader helpers.
    //
    // This line shadows the original `path` variable: before this line it has
    // type `P`; after this line it has type `&Path`.
    let path = path.as_ref();

    // Open and decode the image file. The `?` operator means: if decoding
    // failed, return that error from `read_image_2d`; otherwise unwrap the
    // successful `DynamicImage`.
    let image = decode_image(path)?;

    // `match` branches on the enum value. Rust requires every enum variant to
    // be handled, which prevents accidentally forgetting one channel mode.
    match channels {
        // Auto mode preserves the natural image kind: color sources become
        // three channels, non-color sources become one grayscale channel.
        ImageChannels2D::Auto => {
            // Borrow `image` here because `is_color_image` only needs to inspect
            // it. The image is still owned by this function after the check.
            if is_color_image(&image) {
                // Move the decoded image into the RGB converter and wrap the
                // resulting array in `Ok` to signal success.
                Ok(rgb_image_to_array3(image))
            } else {
                // Move the decoded image into the grayscale converter and wrap
                // the resulting array in `Ok` to signal success.
                Ok(gray_image_to_array3(image))
            }
        }
        // Force grayscale. Even RGB files are converted to `[1, y, x]`.
        ImageChannels2D::Grayscale => Ok(gray_image_to_array3(image)),
        // Force RGB. Even grayscale files are converted to `[3, y, x]`.
        ImageChannels2D::Rgb => Ok(rgb_image_to_array3(image)),
    }
}

/// Read one 2D instance mask as signed labels.
///
/// Background must be encoded as `0`. Positive values are treated as object
/// instance ids. For color masks, use `LabelColorMode2D::ColorIds` or the
/// default auto mode, which converts each unique non-black color into an id.
pub fn read_label_image_2d<P: AsRef<Path>>(
    path: P,
    color_mode: LabelColorMode2D,
) -> Result<Array2<i64>, StarDistTrainError> {
    let path = path.as_ref();
    let image = decode_image(path)?;
    labels_from_dynamic_image(image, color_mode, path)
}

/// Shuffle and split loaded samples into train and validation sets.
///
/// When `validation_fraction > 0` and there is more than one sample, at least
/// one sample is held out for validation.
pub fn split_train_valid_2d(
    mut samples: Vec<TrainingSample2D>,
    validation_fraction: f32,
    seed: u64,
) -> Result<(Vec<TrainingSample2D>, Vec<TrainingSample2D>), StarDistTrainError> {
    if !(0.0..1.0).contains(&validation_fraction) {
        return Err(StarDistTrainError::InvalidConfig(
            "validation_fraction must be in [0, 1)".to_string(),
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    samples.shuffle(&mut rng);

    let len = samples.len();
    let mut valid_count = (len as f32 * validation_fraction).round() as usize;
    if validation_fraction > 0.0 && valid_count == 0 && len > 1 {
        valid_count = 1;
    }
    if valid_count >= len {
        valid_count = len.saturating_sub(1);
    }

    let valid = samples.split_off(len.saturating_sub(valid_count));
    Ok((samples, valid))
}

fn collect_supported_files(
    dir: &Path,
    role: &str,
) -> Result<BTreeMap<String, PathBuf>, StarDistTrainError> {
    if !dir.is_dir() {
        return Err(StarDistTrainError::Dataset(format!(
            "{role} directory does not exist or is not a directory: {}",
            dir.display()
        )));
    }

    let mut files = BTreeMap::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }

        let path = entry.path();
        if !has_supported_extension(&path) {
            continue;
        }

        let stem = file_stem_key(&path)?;
        if let Some(previous) = files.insert(stem.clone(), path.clone()) {
            return Err(StarDistTrainError::Dataset(format!(
                "duplicate {role} file stem '{}': {} and {}",
                stem,
                previous.display(),
                path.display()
            )));
        }
    }

    if files.is_empty() {
        return Err(StarDistTrainError::Dataset(format!(
            "no supported image files found in {role} directory {}; supported extensions: {}",
            dir.display(),
            SUPPORTED_IMAGE_EXTENSIONS_2D.join(", ")
        )));
    }

    Ok(files)
}

fn has_supported_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            let ext = ext.to_ascii_lowercase();
            SUPPORTED_IMAGE_EXTENSIONS_2D.contains(&ext.as_str())
        })
        .unwrap_or(false)
}

fn file_stem_key(path: &Path) -> Result<String, StarDistTrainError> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .map(|stem| stem.to_string())
        .ok_or_else(|| {
            StarDistTrainError::Dataset(format!(
                "file name is not valid UTF-8 or has no stem: {}",
                path.display()
            ))
        })
}

fn decode_image(path: &Path) -> Result<DynamicImage, StarDistTrainError> {
    ImageReader::open(path)?
        .decode()
        .map_err(|source| StarDistTrainError::Image {
            path: path.to_path_buf(),
            source,
        })
}

fn is_color_image(image: &DynamicImage) -> bool {
    matches!(
        image,
        DynamicImage::ImageRgb8(_)
            | DynamicImage::ImageRgb16(_)
            | DynamicImage::ImageRgb32F(_)
            | DynamicImage::ImageRgba8(_)
            | DynamicImage::ImageRgba16(_)
            | DynamicImage::ImageRgba32F(_)
    )
}

fn gray_image_to_array3(image: DynamicImage) -> Array3<f32> {
    let image = image.to_luma32f();
    let (width, height) = image.dimensions();
    let mut out = Array3::<f32>::zeros((1, height as usize, width as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        out[[0, y as usize, x as usize]] = pixel.0[0];
    }
    out
}

fn rgb_image_to_array3(image: DynamicImage) -> Array3<f32> {
    let image = image.to_rgb32f();
    let (width, height) = image.dimensions();
    let mut out = Array3::<f32>::zeros((3, height as usize, width as usize));
    for (x, y, pixel) in image.enumerate_pixels() {
        let yy = y as usize;
        let xx = x as usize;
        out[[0, yy, xx]] = pixel.0[0];
        out[[1, yy, xx]] = pixel.0[1];
        out[[2, yy, xx]] = pixel.0[2];
    }
    out
}

fn labels_from_dynamic_image(
    image: DynamicImage,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    match image {
        DynamicImage::ImageLuma8(buffer) => Ok(labels_from_luma_u8(&buffer)),
        DynamicImage::ImageLuma16(buffer) => Ok(labels_from_luma_u16(&buffer)),
        DynamicImage::ImageLumaA8(buffer) => Ok(labels_from_luma_alpha_u8(&buffer)),
        DynamicImage::ImageLumaA16(buffer) => Ok(labels_from_luma_alpha_u16(&buffer)),
        DynamicImage::ImageRgb8(buffer) => labels_from_rgb_u8(&buffer, color_mode, path),
        DynamicImage::ImageRgba8(buffer) => labels_from_rgba_u8(&buffer, color_mode, path),
        DynamicImage::ImageRgb16(buffer) => labels_from_rgb_u16(&buffer, color_mode, path),
        DynamicImage::ImageRgba16(buffer) => labels_from_rgba_u16(&buffer, color_mode, path),
        DynamicImage::ImageRgb32F(buffer) => labels_from_rgb_f32(&buffer, color_mode, path),
        DynamicImage::ImageRgba32F(buffer) => labels_from_rgba_f32(&buffer, color_mode, path),
        _ => {
            let gray = image.to_luma32f();
            labels_from_luma_f32(&gray, path)
        }
    }
}

fn labels_from_luma_u8(buffer: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Array2<i64> {
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = pixel.0[0] as i64;
    }
    out
}

fn labels_from_luma_u16(buffer: &ImageBuffer<Luma<u16>, Vec<u16>>) -> Array2<i64> {
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = pixel.0[0] as i64;
    }
    out
}

fn labels_from_luma_alpha_u8(buffer: &ImageBuffer<LumaA<u8>, Vec<u8>>) -> Array2<i64> {
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = pixel.0[0] as i64;
    }
    out
}

fn labels_from_luma_alpha_u16(buffer: &ImageBuffer<LumaA<u16>, Vec<u16>>) -> Array2<i64> {
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = pixel.0[0] as i64;
    }
    out
}

fn labels_from_luma_f32(
    buffer: &ImageBuffer<Luma<f32>, Vec<f32>>,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = f32_label_to_i64(pixel.0[0], path)?;
    }
    Ok(out)
}

fn labels_from_rgb_u8(
    buffer: &ImageBuffer<Rgb<u8>, Vec<u8>>,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    let use_color_ids = use_color_ids_u8(
        buffer.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]),
        color_mode,
        path,
    )?;
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        out[[y as usize, x as usize]] = if use_color_ids {
            encode_rgb_u8(r, g, b)
        } else {
            r as i64
        };
    }
    Ok(out)
}

fn labels_from_rgba_u8(
    buffer: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    let use_color_ids = use_color_ids_u8(
        buffer.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]),
        color_mode,
        path,
    )?;
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        out[[y as usize, x as usize]] = if use_color_ids {
            encode_rgb_u8(r, g, b)
        } else {
            r as i64
        };
    }
    Ok(out)
}

fn labels_from_rgb_u16(
    buffer: &ImageBuffer<Rgb<u16>, Vec<u16>>,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    let use_color_ids = use_color_ids_u16(
        buffer.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]),
        color_mode,
        path,
    )?;
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        out[[y as usize, x as usize]] = if use_color_ids {
            encode_rgb_u16(r, g, b)
        } else {
            r as i64
        };
    }
    Ok(out)
}

fn labels_from_rgba_u16(
    buffer: &ImageBuffer<Rgba<u16>, Vec<u16>>,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    let use_color_ids = use_color_ids_u16(
        buffer.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]),
        color_mode,
        path,
    )?;
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        out[[y as usize, x as usize]] = if use_color_ids {
            encode_rgb_u16(r, g, b)
        } else {
            r as i64
        };
    }
    Ok(out)
}

fn labels_from_rgb_f32(
    buffer: &ImageBuffer<Rgb<f32>, Vec<f32>>,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    ensure_f32_labels_are_grayscale(
        buffer.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]),
        color_mode,
        path,
    )?;
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = f32_label_to_i64(pixel.0[0], path)?;
    }
    Ok(out)
}

fn labels_from_rgba_f32(
    buffer: &ImageBuffer<Rgba<f32>, Vec<f32>>,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<Array2<i64>, StarDistTrainError> {
    ensure_f32_labels_are_grayscale(
        buffer.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]),
        color_mode,
        path,
    )?;
    let (width, height) = buffer.dimensions();
    let mut out = Array2::<i64>::zeros((height as usize, width as usize));
    for (x, y, pixel) in buffer.enumerate_pixels() {
        out[[y as usize, x as usize]] = f32_label_to_i64(pixel.0[0], path)?;
    }
    Ok(out)
}

fn use_color_ids_u8<I>(
    pixels: I,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<bool, StarDistTrainError>
where
    I: Iterator<Item = [u8; 3]>,
{
    match color_mode {
        LabelColorMode2D::ColorIds => Ok(true),
        LabelColorMode2D::Auto => Ok(!pixels_are_grayscale(pixels)),
        LabelColorMode2D::Grayscale => {
            if pixels_are_grayscale(pixels) {
                Ok(false)
            } else {
                Err(non_grayscale_label_error(path))
            }
        }
    }
}

fn use_color_ids_u16<I>(
    pixels: I,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<bool, StarDistTrainError>
where
    I: Iterator<Item = [u16; 3]>,
{
    match color_mode {
        LabelColorMode2D::ColorIds => Ok(true),
        LabelColorMode2D::Auto => Ok(!pixels_are_grayscale(pixels)),
        LabelColorMode2D::Grayscale => {
            if pixels_are_grayscale(pixels) {
                Ok(false)
            } else {
                Err(non_grayscale_label_error(path))
            }
        }
    }
}

fn ensure_f32_labels_are_grayscale<I>(
    pixels: I,
    color_mode: LabelColorMode2D,
    path: &Path,
) -> Result<(), StarDistTrainError>
where
    I: Iterator<Item = [f32; 3]>,
{
    if color_mode == LabelColorMode2D::ColorIds {
        return Err(StarDistTrainError::Dataset(format!(
            "floating-point color masks are not supported for {}; use integer color masks or grayscale labels",
            path.display()
        )));
    }

    if pixels_are_grayscale(pixels) {
        Ok(())
    } else {
        Err(non_grayscale_label_error(path))
    }
}

fn pixels_are_grayscale<T, I>(pixels: I) -> bool
where
    T: PartialEq,
    I: Iterator<Item = [T; 3]>,
{
    pixels.into_iter().all(|[r, g, b]| r == g && r == b)
}

fn non_grayscale_label_error(path: &Path) -> StarDistTrainError {
    StarDistTrainError::Dataset(format!(
        "ground-truth mask {} is RGB but channels are not equal; \
         use LabelColorMode2D::ColorIds for color-coded instance masks",
        path.display()
    ))
}

fn encode_rgb_u8(r: u8, g: u8, b: u8) -> i64 {
    (((r as u64) << 16) | ((g as u64) << 8) | b as u64) as i64
}

fn encode_rgb_u16(r: u16, g: u16, b: u16) -> i64 {
    (((r as u64) << 32) | ((g as u64) << 16) | b as u64) as i64
}

fn f32_label_to_i64(value: f32, path: &Path) -> Result<i64, StarDistTrainError> {
    if !value.is_finite() || value < 0.0 {
        return Err(StarDistTrainError::Dataset(format!(
            "ground-truth mask {} contains invalid label value {value}",
            path.display()
        )));
    }

    let rounded = value.round();
    if (value - rounded).abs() > 1.0e-3 {
        return Err(StarDistTrainError::Dataset(format!(
            "ground-truth mask {} contains non-integer label value {value}",
            path.display()
        )));
    }

    Ok(rounded as i64)
}
