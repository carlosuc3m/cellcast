//! Trainable StarDist2D network.
//!
//! The existing `versatile_*_2d` modules are generated inference models that
//! load published pretrained weights by default. This module keeps the same
//! broad architecture shape, but exposes a clean randomly initialized model for
//! training your own instance-segmentation dataset.

use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::prelude::*;

/// Configuration for the trainable 2D StarDist network.
///
/// This first version intentionally follows the architecture already used by
/// `cellcast` pretrained 2D models:
///
/// - input is channel-first `[batch, channel, y, x]`;
/// - the U-Net encoder downsamples by 16 internally;
/// - the decoder upsamples back to prediction grid 1 or 2;
/// - outputs stay channel-first, following Burn's convolution convention.
///
/// Conversion to StarDist/channel-last layout only happens at the inference
/// boundary before calling the existing `cellcast` NMS and label renderer.
#[derive(Clone, Copy, Debug)]
pub struct TrainableStarDist2DConfig {
    /// Number of image channels, for example `1` for fluorescence or `3` for RGB.
    pub n_channel_in: usize,
    /// Number of radial distances predicted for every object center.
    pub n_rays: usize,
    /// Output prediction grid. Supported values are `[1, 1]` and `[2, 2]`.
    pub grid: [usize; 2],
}

impl TrainableStarDist2DConfig {
    /// Create a new network config.
    ///
    /// StarDist polygons need at least three rays. In practice `32` is the
    /// common default and matches the published versatile 2D models.
    pub fn new(n_channel_in: usize, n_rays: usize, grid: [usize; 2]) -> Self {
        Self {
            n_channel_in,
            n_rays,
            grid,
        }
    }

    /// Initialize a randomly weighted model on the selected Burn device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrainableStarDist2D<B> {
        TrainableStarDist2D::new(self.n_channel_in, self.n_rays, self.grid, device)
    }
}

/// A U-Net-like StarDist2D model with one probability head and one distance head.
///
/// Burn models are regular Rust structs. The `Module` derive tells Burn which
/// fields contain trainable parameters so optimizers can update them.
#[derive(Module, Debug)]
pub struct TrainableStarDist2D<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d3: Conv2d<B>,
    conv2d4: Conv2d<B>,
    maxpool2d2: MaxPool2d,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    maxpool2d3: MaxPool2d,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    maxpool2d4: MaxPool2d,
    conv2d9: Conv2d<B>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    full_conv1: Conv2d<B>,
    full_conv2: Conv2d<B>,
    conv2d17: Conv2d<B>,
    dist_head: Conv2d<B>,
    prob_head: Conv2d<B>,
    grid: [usize; 2],
}

impl<B: Backend> TrainableStarDist2D<B> {
    /// Build a randomly initialized StarDist2D model.
    pub fn new(n_channel_in: usize, n_rays: usize, grid: [usize; 2], device: &B::Device) -> Self {
        let conv2d1 = conv(n_channel_in, 32, 3, device);
        let conv2d2 = conv(32, 32, 3, device);
        let maxpool2d1 = maxpool2();
        let conv2d3 = conv(32, 32, 3, device);
        let conv2d4 = conv(32, 32, 3, device);
        let maxpool2d2 = maxpool2();
        let conv2d5 = conv(32, 64, 3, device);
        let conv2d6 = conv(64, 64, 3, device);
        let maxpool2d3 = maxpool2();
        let conv2d7 = conv(64, 128, 3, device);
        let conv2d8 = conv(128, 128, 3, device);
        let maxpool2d4 = maxpool2();
        let conv2d9 = conv(128, 256, 3, device);
        let conv2d10 = conv(256, 128, 3, device);
        let conv2d11 = conv(256, 128, 3, device);
        let conv2d12 = conv(128, 64, 3, device);
        let conv2d13 = conv(128, 64, 3, device);
        let conv2d14 = conv(64, 32, 3, device);
        let conv2d15 = conv(64, 32, 3, device);
        let conv2d16 = conv(32, 32, 3, device);
        let full_conv1 = conv(64, 32, 3, device);
        let full_conv2 = conv(32, 32, 3, device);
        let conv2d17 = conv(32, 128, 3, device);
        let dist_head = conv1x1(128, n_rays, device);
        let prob_head = conv1x1(128, 1, device);

        Self {
            conv2d1,
            conv2d2,
            maxpool2d1,
            conv2d3,
            conv2d4,
            maxpool2d2,
            conv2d5,
            conv2d6,
            maxpool2d3,
            conv2d7,
            conv2d8,
            maxpool2d4,
            conv2d9,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            full_conv1,
            full_conv2,
            conv2d17,
            dist_head,
            prob_head,
            grid,
        }
    }

    /// Run the model.
    ///
    /// Input shape is `[batch, channel, y, x]`. Both spatial dimensions must be
    /// divisible by 16 because the encoder pools four times. Output shapes are:
    ///
    /// For grid `[2, 2]`, outputs are:
    ///
    /// - probability: `[batch, 1, y / 2, x / 2]`
    /// - distances: `[batch, n_rays, y / 2, x / 2]`
    ///
    /// For grid `[1, 1]`, outputs are:
    ///
    /// - probability: `[batch, 1, y, x]`
    /// - distances: `[batch, n_rays, y, x]`
    #[allow(clippy::let_and_return)]
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let relu2 = {
            let x = self.conv2d1.forward(input);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d2.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu4 = {
            let x = self.maxpool2d1.forward(relu2.clone());
            let x = self.conv2d3.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d4.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu6 = {
            let x = self.maxpool2d2.forward(relu4.clone());
            let x = self.conv2d5.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d6.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu8 = {
            let x = self.maxpool2d3.forward(relu6.clone());
            let x = self.conv2d7.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d8.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu10 = {
            let x = self.maxpool2d4.forward(relu8.clone());
            let x = self.conv2d9.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d10.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu12 = {
            let x = upsample_nearest_2x(relu10);
            let x = Tensor::cat([x, relu8].into(), 1);
            let x = self.conv2d11.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d12.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu14 = {
            let x = upsample_nearest_2x(relu12);
            let x = Tensor::cat([x, relu6].into(), 1);
            let x = self.conv2d13.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d14.forward(x);
            burn::tensor::activation::relu(x)
        };

        let relu16 = {
            let x = upsample_nearest_2x(relu14);
            let x = Tensor::cat([x, relu4].into(), 1);
            let x = self.conv2d15.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.conv2d16.forward(x);
            burn::tensor::activation::relu(x)
        };

        let features = if self.grid == [1, 1] {
            let x = upsample_nearest_2x(relu16);
            let x = Tensor::cat([x, relu2].into(), 1);
            let x = self.full_conv1.forward(x);
            let x = burn::tensor::activation::relu(x);
            let x = self.full_conv2.forward(x);
            burn::tensor::activation::relu(x)
        } else {
            relu16
        };

        let features = self.conv2d17.forward(features);
        let features = burn::tensor::activation::relu(features);

        let dist = self.dist_head.forward(features.clone());
        let prob = self.prob_head.forward(features);
        let prob = burn::tensor::activation::sigmoid(prob);

        (prob, dist)
    }
}

fn conv<B: Backend>(
    channels_in: usize,
    channels_out: usize,
    kernel: usize,
    device: &B::Device,
) -> Conv2d<B> {
    let pad = kernel / 2;
    Conv2dConfig::new([channels_in, channels_out], [kernel, kernel])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(pad, pad))
        .with_dilation([1, 1])
        .with_groups(1)
        .with_bias(true)
        .init(device)
}

fn conv1x1<B: Backend>(channels_in: usize, channels_out: usize, device: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([channels_in, channels_out], [1, 1])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Valid)
        .with_dilation([1, 1])
        .with_groups(1)
        .with_bias(true)
        .init(device)
}

fn maxpool2() -> MaxPool2d {
    MaxPool2dConfig::new([2, 2])
        .with_strides([2, 2])
        .with_padding(PaddingConfig2d::Valid)
        .with_dilation([1, 1])
        .init()
}

/// Nearest-neighbor 2x upsampling implemented with tensor reshape/repeat ops.
///
/// Burn 0.20 supports the low-level tensor operations we need here, and this
/// avoids introducing another layer type while keeping behavior identical to
/// the generated `cellcast` StarDist inference graphs.
fn upsample_nearest_2x<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    let batch = batch as i32;
    let channels = channels as i32;
    let height = height as i32;
    let width = width as i32;

    let x: Tensor<B, 5> = input.unsqueeze_dims(&[3]);
    let x = x.repeat(&[1, 1, 1, 2, 1]);
    let x = x.permute([0, 2, 3, 4, 1]);
    let x = x.reshape([batch, height * 2, width, channels]);
    let x: Tensor<B, 5> = x.unsqueeze_dims(&[3]);
    let x = x.repeat(&[1, 1, 1, 2, 1]);
    let x = x.reshape([batch, height * 2, width * 2, channels]);
    x.permute([0, 3, 1, 2])
}
