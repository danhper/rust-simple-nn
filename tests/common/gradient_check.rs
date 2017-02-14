use simple_nn::Network;

use common::fixtures;

pub fn check_gradients(network: &Network) {
    let X = fixtures::load_matrix("mnist_sample.txt");
    let Y = fixtures::load_matrix("mnist_sample_labels.txt").to_one_hot(10);

    let results = network.forward(&X);
    let gradients = network.backward(&results, &Y);

    let epsilon = 0.0001;

    network.iterate_layers(|layer, layer_index| {
        let ref backprop_grads = gradients[layer_index];
    });
}
