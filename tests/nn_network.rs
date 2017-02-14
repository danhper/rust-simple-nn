extern crate simple_nn;

mod common;
use common::fixtures;

use simple_nn::{layers, objectives, Network, Matrix, OutputLayer};

#[test]
fn network_add() {
    let mut network = Network::new();
    assert_eq!(network.layers_count(), 0);
    network.add(layers::Dense::new(10, 10));
    assert_eq!(network.layers_count(), 1);
}

#[test]
fn network_add_final() {
    let mut network = Network::new();
    network.add(layers::Dense::new(10, 10));
    network.add(layers::Relu::new());
    network.add_final(layers::Softmax::new().minimizing(objectives::CrossEntropy::new()));
    assert_eq!(network.layers_count(), 3);
}

#[test]
fn network_forward() {
    let input = Matrix::new_from(2, 3, vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]);
    let weights = Matrix::new_from(3, 3, vec![2.0, 4.0, 5.0, -8.0, 3.0, -1.0, 7.0, -2.0, 6.0]);

    let mut network = Network::new();
    network.add(layers::Dense::new_with_weights(&weights));
    network.add(layers::Relu::new());
    // network.add_final(layers::Softmax::new()::minimizing(objectives::CrossEntropy));

    let results = network.forward(&input);
    assert_eq!(results[0], input);
    assert_eq!(results[1], Matrix::new_from(2, 3, vec![39.0, -8.0, 25.0, -74.0, 43.0, -21.0]));
    assert_eq!(results[2], Matrix::new_from(2, 3, vec![39.0, 0.0, 25.0, 0.0, 43.0, 0.0]));
}

#[test]
fn network_backward() {
    let mut network = Network::new();
    network.add(layers::Dense::new(784, 100));
    network.add(layers::Relu::new());
    network.add(layers::Dense::new(100, 10));
    network.add_final(layers::Softmax::new().minimizing(objectives::CrossEntropy::new()));
    common::check_gradients(&network)
}
