use std::cmp;

use nn::{layers, objectives, optimizers, functions};
use nn::formatter::Formatter;
use nn::training_results::TrainingResults;
use linalg::{Matrix};

pub struct TrainOptions {
    pub shuffle: bool,
    pub epochs: usize,
    pub batch_size: usize
}

impl TrainOptions {
    pub fn default() -> TrainOptions {
        TrainOptions {
            shuffle: true,
            epochs: 1,
            batch_size: 64
        }
    }

    pub fn with_epochs(mut self, epochs: usize) -> TrainOptions {
        self.epochs = epochs;
        self
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> TrainOptions {
        self.shuffle = shuffle;
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> TrainOptions {
        self.batch_size = batch_size;
        self
    }
}

pub struct Network<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone> {
    layers: Vec<Box<layers::Layer>>,
    objective: Obj,
    optimizer: Opt,
    output: Box<Out>,
    formatter: Formatter
}

impl<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone> Network<Out, Obj, Opt> {
    pub fn new(layers: Vec<Box<layers::Layer>>, objective: Obj,
               optimizer: Opt, output: Box<Out>, formatter: Formatter) -> Network<Out, Obj, Opt> {
        Network {
            layers: layers,
            objective: objective,
            optimizer: optimizer,
            output: output,
            formatter: formatter
        }
    }

    pub fn layers_count(&self) -> usize {
        self.layers.len()
    }

    pub fn get_layer(&self, index: usize) -> &Box<layers::Layer> {
        &self.layers[index]
    }

    pub fn get_mut_layer(&mut self, index: usize) -> &mut Box<layers::Layer> {
        &mut self.layers[index]
    }

    pub fn fit(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, train_options: TrainOptions) {
        for i in 0..train_options.epochs {
            println!("\nTraining epoch {} / {}", i + 1, train_options.epochs);
            self.process_and_run_epoch(input, expected, &train_options);
        }
    }

    fn process_and_run_epoch(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, train_options: &TrainOptions) {
        if train_options.shuffle {
            let mut cloned_input = input.clone();
            let mut cloned_expected = expected.clone();
            self.shuffle(&mut cloned_input, &mut cloned_expected);
            self.run_epoch(&cloned_input, &cloned_expected, train_options);
        } else {
            self.run_epoch(input, expected, train_options)
        }
    }

    fn run_epoch(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>, train_options: &TrainOptions) {
        let total_batches = (input.rows / train_options.batch_size) + ((input.rows % train_options.batch_size != 0) as usize);
        let mut results = TrainingResults::default();
        results.total_count = input.rows as u64;
        for n in 0..total_batches {
            let start = n * train_options.batch_size;
            let end = cmp::min(train_options.batch_size * (n + 1), input.rows);
            let x = input.slice_rows(start..end);
            let y = expected.slice_rows(start..end);

            let (hit_count, miss_count, loss) = self.train_on_batch(&x, &y);
            // TODO: use callbacks or something to handle this
            self.update_result(&mut results, (end - start) as u64, hit_count, miss_count, loss);
            print!("{}\r", self.formatter.progress(&results))
        }
    }

    fn update_result(&self, results: &mut TrainingResults, count: u64, hit_count: u64, miss_count: u64, loss: f64) {
        results.current_count += count;
        results.hit_count += hit_count;
        results.miss_count += miss_count;
        results.total_loss += loss;
    }

    pub fn shuffle<T: Clone>(&self, input: &mut Matrix<T>, expected: &mut Matrix<T>) {
        let swaps = input.shuffle_rows();
        for (row, other) in swaps {
            expected.swap_rows(row, other);
        }
    }

    pub fn train_on_batch(&mut self, input: &Matrix<f64>, expected: &Matrix<f64>) -> (u64, u64, f64) {
        let results = self.forward(input);
        let gradients = self.backward(&results, expected);
        let ref optimizer = self.optimizer.clone();
        for (index, gradient) in gradients {
            let mut weights = self.get_mut_layer(index).get_mut_weights();
            optimizer.apply_gradients(weights, &gradient, input.rows);
        }
        let last = results.last().unwrap();
        let loss = self.loss_from_probs(&last, expected);
        let (hit_count, miss_count) = functions::hit_miss_from_probs(&last, expected);
        (hit_count, miss_count, loss)
    }

    pub fn predict_probs(&self, input: &Matrix<f64>) -> Matrix<f64> {
        let results = self.forward(input);
        let output = results.last().unwrap();
        output.clone()
    }

    pub fn predict(&self, input: &Matrix<f64>) -> Matrix<usize> {
        let probs = self.predict_probs(input);
        functions::argmax(&probs)
    }

    pub fn accuracy<T: From<u8> + Clone + PartialEq>(&self, input: &Matrix<f64>, expected: &Matrix<T>) -> f64 {
        let probs = self.predict_probs(input);
        functions::accuracy_from_probs(&probs, expected)
    }

    pub fn accuracy_from_probs<T: From<u8> + Clone + PartialEq>(&self, probs: &Matrix<f64>, expected: &Matrix<T>) -> f64 {
        functions::accuracy_from_probs(&probs, expected)
    }

    pub fn mean_loss(&self, input: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        self.loss(input, expected) / (input.rows as f64)
    }

    pub fn loss(&self, input: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        let probs = self.predict_probs(input);
        self.loss_from_probs(&probs, expected)
    }

    pub fn mean_loss_from_probs(&self, predictions: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        self.loss_from_probs(predictions, expected) / (predictions.rows as f64)
    }

    pub fn loss_from_probs(&self, predictions: &Matrix<f64>, expected: &Matrix<f64>) -> f64 {
        self.objective.loss(&predictions, expected).reduce(0.0, |acc, v| acc + v)
    }

    pub fn forward(&self, input: &Matrix<f64>) -> Vec<Matrix<f64>> {
        let mut results = vec![input.clone()];
        for layer in self.layers.iter() {
            let next = layer.compute(&results.last().unwrap());
            results.push(next);
        }
        let next = self.output.compute(&results.last().unwrap());
        results.push(next);
        results
    }

    pub fn backward(&self, results: &Vec<Matrix<f64>>, expected: &Matrix<f64>) -> Vec<(usize, Matrix<f64>)> {
        let mut gradients: Vec<(usize, Matrix<f64>)> = vec![];
        let mut back_results = vec![self.objective.delta(&results[results.len() - 1], expected)];
        let last_layer_index = self.layers_count();
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
