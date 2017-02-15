use rand;
use rand::distributions::{IndependentSample, Range};

use simple_nn::{Network, objectives, optimizers, layers};
use common::fixtures;

fn get_layer_dim<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone>(network: &Network<Out, Obj, Opt>, layer_index: usize) -> (usize, usize) {
    let layer = network.get_layer(layer_index);
    if layer.has_trainable_weights() {
        let weights = layer.get_weights();
        (weights.rows, weights.columns)
    } else {
        (0, 0)
    }
}

fn get_weight<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone>(network: &Network<Out, Obj, Opt>, layer_index: usize, row: usize, col: usize) -> f64 {
    network.get_layer(layer_index).get_weights().at(row, col)
}

fn set_weight<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone>(network: &mut Network<Out, Obj, Opt>, layer_index: usize, row: usize, col: usize, val: f64) {
    network.get_mut_layer(layer_index).get_mut_weights().set_at(row, col, val);
}

#[allow(dead_code)]
pub fn check_gradients<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone>(network: &mut Network<Out, Obj, Opt>) {
    let check_count = 50;
    let x = fixtures::load_matrix("mnist_sample.txt").transform(|x: f64| x / 255.0);
    let y = fixtures::load_matrix("mnist_sample_labels.txt").to_one_hot(10);

    let results = network.forward(&x);
    let gradients = network.backward(&results, &y);
    let epsilon = 0.0001;

    let mut rng = rand::thread_rng();

    for (layer_index, backprop_grads) in gradients {
        let (rows, columns) = get_layer_dim(network, layer_index);

        let row_range = Range::new(0, rows);
        let col_range = Range::new(0, columns);

        for _ in 0..check_count {
            let row = row_range.ind_sample(&mut rng);
            let col = col_range.ind_sample(&mut rng);

            let original = get_weight(network, layer_index, row, col);
            set_weight(network, layer_index, row, col, original + epsilon);
            let plus_cost = network.loss(&x, &y);
            set_weight(network, layer_index, row, col, original - epsilon);
            let minus_cost = network.loss(&x, &y);
            set_weight(network, layer_index, row, col, original);

            let grad = (plus_cost - minus_cost) / (2.0 * epsilon);
            let backprop_grad = backprop_grads.at(row, col);

            let diff = (grad - backprop_grad).abs();
            assert!(diff < 1e-5, "gradient check failed, numerical: {}, backprop: {}", grad, backprop_grad);
        }
    }
}
