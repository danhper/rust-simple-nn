#![feature(box_syntax, box_patterns)]

extern crate rand;

pub use linalg::matrix::{Matrix};
pub use nn::{layers, objectives, optimizers, Network};
pub use nn::output_layers::OutputLayer;

pub mod linalg;
pub mod nn;
pub mod utils;
