extern crate rand;
extern crate simple_nn;

mod common;

use simple_nn::{layers, objectives, optimizers, NetworkBuilder, Matrix};

#[test]
fn network_builder_add() {
    let network = NetworkBuilder::new()
        .add(layers::Dense::new(10, 10))
        .add_output(layers::Softmax::new())
        .minimize(objectives::CrossEntropy::new())
        .with(optimizers::SGD::new(0.5))
        .build();
    assert_eq!(network.layers_count(), 1);
}

#[test]
fn network_forward() {
    let input = Matrix::new_from(2, 3, vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0], true);
    let weights = Matrix::new_from(3, 3, vec![2.0, 4.0, 5.0, -8.0, 3.0, -1.0, 7.0, -2.0, 6.0], true);

    let network = NetworkBuilder::new()
        .add(layers::Dense::new_with_weights(&weights))
        .add(layers::Relu::new())
        .add_output(layers::Softmax::new())
        .minimize(objectives::CrossEntropy::new())
        .with(optimizers::SGD::new(0.5))
        .build();

    let results = network.forward(&input);
    assert_eq!(results[0], input);
    assert_eq!(results[1], Matrix::new_from(2, 3, vec![39.0, -8.0, 25.0, -74.0, 43.0, -21.0], true));
    assert_eq!(results[2], Matrix::new_from(2, 3, vec![39.0, 0.0, 25.0, 0.0, 43.0, 0.0], true));
}

#[test]
#[ignore]
fn network_backward() {
    let mut network = NetworkBuilder::new()
        .add(layers::Dense::new(784, 100))
        .add(layers::Relu::new())
        .add(layers::Dense::new(100, 100))
        .add(layers::Relu::new())
        .add(layers::Dense::new(100, 10))
        .add_output(layers::Softmax::new())
        .minimize(objectives::CrossEntropy::new())
        .with(optimizers::SGD::new(0.5))
        .build();
    common::check_gradients(&mut network)
}
