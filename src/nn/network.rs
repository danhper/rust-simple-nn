use nn::{layers, output_layers, objectives};
use linalg::{Matrix};

pub struct Network {
    layers: Vec<Box<layers::Layer>>,
    objective: Option<Box<objectives::Objective>>
}

impl Network {
    pub fn new() -> Network {
        Network {
            layers: vec![],
            objective: None
        }
    }

    pub fn layers_count(&self) -> usize {
        self.layers.len()
    }

    pub fn add(&mut self, layer: Box<layers::Layer>) {
        self.layers.push(layer)
    }

    pub fn iterate_layers<F>(&self, mut f: F)
            where F: FnMut(&layers::Layer, usize) {
        for i in 0..self.layers.len() {
            f(self.layers[i].as_ref(), i)
        }
    }

    pub fn add_final(&mut self, final_layer: output_layers::FinalLayer) {
        self.objective = Some(final_layer.objective);
        self.add(final_layer.layer)
    }

    pub fn predict(&self, input: &Matrix) -> Matrix {
        let results = self.forward(input);
        results.last().unwrap().clone()
    }

    pub fn score(&self, input: &Matrix, expected: &Matrix) -> f64 {
        let predictions = self.predict(input);
        self.objective.as_ref().unwrap().loss(&predictions, expected).reduce(0.0, |acc, v| acc + v)
    }

    pub fn forward(&self, input: &Matrix) -> Vec<Matrix> {
        let mut results = vec![input.clone()];
        for layer in self.layers.iter() {
            let next = layer.compute(&results.last().unwrap());
            results.push(next);
        }
        results
    }

    pub fn backward(&self, results: &Vec<Matrix>, expected: &Matrix) -> Vec<Matrix> {
        let objective = self.objective.as_ref().unwrap();
        let mut gradients = vec![objective.delta(&results[results.len() - 1], expected)];
        let last_layer_index = self.layers.len() - 2;
        for i in (1..last_layer_index).rev() {
            let gradient = self.layers[i].delta(&results[i], &results[i + 1], &gradients[gradients.len() - 1]);
            gradients.push(gradient);
        }
        gradients.reverse();
        gradients
    }
}
