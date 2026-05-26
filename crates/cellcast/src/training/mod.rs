//! Training entry points.
//!
//! The first production target is 2D single-class StarDist training. The data
//! and tensor layout in this module follows Burn conventions: images and model
//! outputs are channel-first.

pub mod io_2d;
pub mod stardist_2d;
