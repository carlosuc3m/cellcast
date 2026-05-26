# cellcast: A recast of cell segmentation models

<div align="center">

[![crates.io](https://img.shields.io/crates/v/cellcast)](https://crates.io/crates/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

This crate contains the [cellcast](https://github.com/uw-loci/cellcast) core Rust library. Cellcast is a recast of cell segmentation models
built on the Burn tensor and deep learning framework. The goal of this project is to modernize (*i.e.* recast) established cell segmentation models
with a WebGPU backend. Cellcast aims to make access to cell segmentation models **easy** and **reproducible**.

## Usage

### Using cellcast with Rust

To use cellcast in your Rust project add it to your crate's dependencies and import the desired models.

```toml
[dependencies]
cellcast = "0.2.0"
```

The example below demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with Rust.
This example assumes you have the appropriate dependencies and helper functions to load your data as an `Array2<T>` type:

```rust
use ndarray::Array2;
use cellcast::models::stardist_2d::predict_versatile_fluo;

fn main() {
  let data_2d = load_image("/path/to/data_2d.tif");
  let labels = predict_versatile_fluo(&data, Some(1.0), Some(99.8), None, None, True);
}

fn load_image(path: &str) -> Array2<u16> {
  // your logic to read/load from a file here
}
```

*Note: `T` here can be any numeric value (*i.e.* `u8`, `i32`, `f64`).*

### Training StarDist2D

The trainable StarDist2D path is available from `cellcast::training::stardist_2d`.
Training data uses Burn's channel-first convention: images are `Array3<f32>` with
shape `[channel, y, x]`, and instance labels are `Array2<i64>` with shape
`[y, x]`. Negative label values mark pixels that should be ignored by the
probability loss, matching the original StarDist training convention. The
WebGPU entry point uses Burn's `Wgpu` backend, so it can run on supported
NVIDIA, AMD, Intel, and Apple GPUs. CPU training is available through the
`NdArray` backend.

```rust
use cellcast::training::stardist_2d::{
    load_stardist_2d, predict_stardist_2d, save_stardist_2d, train_stardist_2d_wgpu,
    TrainingConfig2D, TrainingSample2D, WgpuInferBackend,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let train: Vec<TrainingSample2D> = load_training_samples();
    let valid: Vec<TrainingSample2D> = load_validation_samples();

    let config = TrainingConfig2D {
        n_channel_in: 1,
        n_rays: 32,
        grid: [1, 1],
        patch_size: [256, 256],
        batch_size: 4,
        ..Default::default()
    };

    let result = train_stardist_2d_wgpu(&train, Some(valid.as_slice()), config)?;
    save_stardist_2d(result.model, &result.config, "artifacts/stardist2d")?;

    let device = Default::default();
    let (model, config) = load_stardist_2d::<WgpuInferBackend, _>(
        "artifacts/stardist2d",
        &device,
    )?;
    let labels = predict_stardist_2d(
        &model,
        &valid[0].image,
        &config,
        &device,
    )?;

    Ok(())
}
```

For a folder dataset, put input images in one folder and instance masks in
another folder with matching file stems:

```text
dataset/
  data/
    img_001.tif
    img_002.png
  gt/
    img_001.tif
    img_002.png
```

Then run the included example:

```bash
cargo run --release -p cellcast --example train_stardist_2d_folder -- \
  dataset/data dataset/gt artifacts/stardist2d \
  --epochs 400 --steps 100 --batch 4 --patch 256 --grid 1 \
  --normalize-percentile
```

For auto-detected dataset roots, omit the GT folder:

```bash
cargo run --release -p cellcast --example train_stardist_2d_folder -- \
  dataset artifacts/stardist2d \
  --epochs 400 --steps 100 --batch 4 --patch 256 --grid 1 \
  --normalize-percentile
```

The loader accepts `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`, `.bmp`, and PNM
files. The masks must be instance labels with `0` as background. Grayscale masks
are read as integer label ids; color masks can be read as color-coded instance
ids by adding `--label-color-ids`. Add `--cpu` to train with the CPU backend
instead of WebGPU.

The Rust config defaults to original-StarDist normalization semantics: images
are assumed to be normalized before training. Set `normalization` to percentile
normalization explicitly if you want the Rust trainer to normalize crops.

The training loop includes the original-style `ReduceOnPlateau` learning-rate
schedule by default, plus `Constant`, `CosineAnnealing`, and `PolynomialDecay`
options. `optimize_thresholds_stardist_2d` can tune probability and NMS
thresholds on validation images after training.

Shape completion is available through `shape_completion = true`. In that mode,
the trainer samples a larger label patch but feeds only the center crop to the
network. Objects touching the outer patch border are removed from the distance
target, so the model learns to predict complete shapes for objects that are
only partially visible in the input crop.

The trainable architecture supports StarDist grids `[1, 1]` and `[2, 2]`.
Grid `[1, 1]` predicts at every pixel; grid `[2, 2]` predicts every second
pixel and is lighter. The network and losses stay channel-first; conversion to
StarDist's `[y, x, rays]` layout only happens when calling the polygon NMS and
label renderer.

## Building from source

You can build the cellcast core library with:

```bash
$ cargo build
```

This will compile a cellcast *without optimizations*. Pass the `--release` flag to compile an *optimized* release version (note that compilation time may take upwards
of 10 minutes). Because cellcast is a library, compiling it on it's own isn't very useful. However being able to successfully compile cellcast on your own computer
means that you can change the backend from `Wgpu` to whatever other [supported Burn backend](https://github.com/Tracel-AI/burn?tab=readme-ov-file#supported-backends)
you want. Recompiling cellcast with a *different* backend may allow you to take advantage of hardware specific optimizations not available to the `Wgpu` backend.

The release version of cellcast uses the `NdArrayBackend` and `WgpuBackend` for CPU and GPU compute respectively. The CPU and GPU backends are defined in the `backend.rs`
file in the `config` module. 

```rust
pub(crate) type CpuBackend<E, I> = NdArray<E, I>;
pub(crate) type GpuBackend<E, I> = Wgpu<E, I>;
```

Change the `Wgpu` and/or `NdArray` Burn backends here and recompile cellcast to change the project's backend.

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](cellcast/MODEL-LICENSES) file.
