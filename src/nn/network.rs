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

    pub fn get_layer(&self, index: usize) -> &Box<layers::Layer> {
        &self.layers[index]
    }

    pub fn get_mut_layer(&mut self, index: usize) -> &mut Box<layers::Layer> {
        &mut self.layers[index]
    }

    pub fn add_final(&mut self, final_layer: output_layers::FinalLayer) {
        self.objective = Some(final_layer.objective);
        self.add(final_layer.layer)
    }

    pub fn predict(&self, input: &Matrix) -> Matrix {
        let results = self.forward(input);
        let output = results.last().unwrap();
        output.clone()
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

    pub fn backward(&self, results: &Vec<Matrix>, expected: &Matrix) -> Vec<(usize, Matrix)> {
        let objective = self.objective.as_ref().unwrap();
        let mut gradients: Vec<(usize, Matrix)> = vec![];
        let mut back_results = vec![objective.delta(&results[results.len() - 1], expected)];
        let last_layer_index = self.layers_count() - 1;
        for i in (0..last_layer_index).rev() {
            let gradient = self.layers[i].delta(&results[i], &results[i + 1], &back_results[back_results.len() - 1]);
            if self.layers[i].has_trainable_weights() {
                let gradient = results[i].t().matmul(&back_results[back_results.len() - 1]);
                gradients.push((i, gradient));
            }
            back_results.push(gradient);
        }
        gradients.reverse();
        gradients
    }
}
