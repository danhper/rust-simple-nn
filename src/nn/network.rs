use nn::layers::Layer;
use linalg::{Matrix};

#[allow(dead_code)]
pub struct Network {
    layers: Vec<Box<Layer>>
}

#[allow(dead_code)]
impl Network {
    pub fn new() -> Network {
        Network {
            layers: vec![]
        }
    }

    pub fn add(&mut self, layer: Box<Layer>) {
        self.layers.push(layer)
    }

    pub fn forward(&self, input: &Matrix) -> Vec<Matrix> {
        let mut results = vec![input.clone()];
        for layer in self.layers.iter() {
            let next = layer.compute(&results.last().unwrap());
            results.push(next);
        }
        results
    }
}
