# cellcast_python

<div align="center">

[![pypi](https://img.shields.io/pypi/v/cellcast)](https://pypi.org/project/cellcast)
![license](https://img.shields.io/badge/license-MIT/Unlicense-blue)

</div>

This crate contains the Python bindings (via PyO3) for the [cellcast](https://github.com/uw-loci/cellcast)
core Rust library. Cellcast is a recast of cell segmentation models built on the Burn tensor and deep
learning framework. The goal of this project is to modernize (*i.e.* recast) established cell segmentation models
with a WebGPU backend. Cellcast aims to make access to cell segmentation models **easy** and **reproducible**.

## Installation

### Requirements

The `cellcast` Python package currently supports the following architectures:

| Operating System | Architecture         |
| :---             | :---                 |
| Linux            | x86-64, arm64        |
| macOS            | intel, arm64         |
| Windows          | x86-64               |

Cellcast is compatible with Python `>=3.7` and requires *only* `NumPy`.

### cellcast from PyPI

You can install the cellcast Python package from PyPI with:

```bash
$ pip install cellcast
```

### Build cellcast_python from source

To build the cellcat_python package from source, use the `maturin` build tool
(this requires the Rust toolchain). If you're using `uv` to manage your Python
virtual environments (venv) add `numpy` and `maturin` to your environment and run the
`maturin develop` command in the `cellcast_python` directory of the
[cellcast](https://github.com/uw-loci/cellcast) repository with your venv activated:

```bash
$ source ~/path/to/myenv/.venv/bin/activate
$ (myenv) cd cellcast_python
$ maturin develop
```

Alernatively if you're using `conda` or `mamba` you can do the following:

```bash
$ cd cellcat_python
$ mamba activate myenv
(myenv) $ mamba install numpy maturin
...
(myenv) $ maturin develop
```

This will compile a *non-optimized* cellcast binaries. Pass the `--release` flag to
compile optimized binaries (note that compilation time may take upwards of 10 minutes).

### Build reusable wheels in GitHub Actions

The `Build Python Wheels` workflow builds installable Python wheels for Linux,
Windows, and macOS. Run it from GitHub's Actions tab or push to the
`stardist-burn-training` branch. The combined `cellcast-wheelhouse` artifact can
be downloaded and copied into another project.

To install from a local wheelhouse without requiring Rust on the target machine:

```bash
python -m pip install --no-index --find-links ./vendor/wheels cellcast==0.2.1.dev0
```

## Usage

### Using cellcast

Once cellcast has been installed, `cellcast` will be available to import. The example below
demonstrates how to use cellcast and the StarDist 2D versatile fluo segmentation model with
Python. Note that this example assumes you have access to 2D data and `tifffile` installed
in your Python environment with cellcast:

```python
import cellcast.models.stardist_2d as sd
from tifffile import imread

# load 2D data for inference
data_2d = imread("path/to/data_2d.tif")

# run stardist inference and produce instance segmentations
labels = sd.predict_versatile_fluo(data, gpu=True)
```

Run `help()` on the `predict_versatile_fluo()` function to see the full function signature and default values.

### Training StarDist2D from Python

The training API reads paired image and ground-truth folders and can report
progress through Python callbacks:

```python
import cellcast.training.stardist_2d as train


def on_train_begin(plan):
    print(f"training {plan['epochs']} epochs, {plan['total_steps']} steps")


def on_step_end(step):
    print(step["epoch"], step["step"], step["loss_total"])


def on_validation_end(epoch):
    print(epoch["epoch"], epoch["train_total"], epoch["valid_total"])
    for preview in epoch["previews"]:
        image = preview["image"]
        labels = preview["labels"]
        prediction = preview["prediction"]
        prob = preview["prob"]
        # Display these arrays in your UI/notebook.


result = train.train_stardist_2d_folder(
    "dataset/data",
    "dataset/gt",
    output_dir="artifacts/stardist2d",
    gpu=False,
    image_channels="grayscale",
    config={
        "epochs": 100,
        "steps_per_epoch": 100,
        "validation_steps": 10,
        "batch_size": 4,
        "patch_size": [256, 256],
        "grid": [1, 1],
        "n_rays": 32,
        "validation_preview_count": 2,
    },
    on_train_begin=on_train_begin,
    on_step_end=on_step_end,
    on_validation_end=on_validation_end,
)
```

`on_train_begin` receives the planned epochs, steps, batch size, patch size, and
sample counts. `on_step_end` receives the current total, probability, and
distance losses after every optimizer step. `on_validation_end` receives epoch
train/validation losses and, when `validation_preview_count > 0`, NumPy arrays
for validation image, ground truth, predicted labels, and probability map.

## License

Cellcast *itself* is a dual-licensed project with your choice of:

- MIT License (see [LICENSE-MIT](../LICENSE-MIT))
- The Unlicense (see [LICENSE-UNLICENSE](../LICENSE-UNLICENSE))

These licenses only apply to the cellcast project and **do not** apply to the individual models supported
by cellcast. You can find each model's associated license listed in the [MODEL-LICENSES](../cellcast/MODEL-LICENSES) file.
