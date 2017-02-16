pub use nn::network::{Network, TrainOptions};
pub use nn::network_builder::{NetworkBuilder};

pub use nn::formatter::Formatter;

pub mod layers;
pub mod network;
pub mod network_builder;
pub mod functions;
pub mod objectives;
pub mod optimizers;
pub mod training_results;
pub mod measures;
pub mod formatter;
