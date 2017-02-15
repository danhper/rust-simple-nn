extern crate rand;

pub use linalg::matrix::{Matrix};
pub use nn::{layers, objectives, optimizers, Network, NetworkBuilder};

pub mod linalg;
pub mod nn;
pub mod utils;
