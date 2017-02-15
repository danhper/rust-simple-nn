extern crate simple_nn;

use simple_nn::{nn, utils, OutputLayer};

fn main() {
    let mut network = nn::Network::new();
    network.add(nn::layers::Dense::new(784, 100));
    network.add(nn::layers::Relu::new());
    network.add(nn::layers::Dense::new(100, 100));
    network.add(nn::layers::Relu::new());
    network.add(nn::layers::Dense::new(100, 10));
    network.add_final(nn::layers::Softmax::new().minimizing(nn::objectives::CrossEntropy::new()));
    network.optimize_with(nn::optimizers::SGD::new(0.01));

    let x_train = utils::loader::matrix_from_txt("tests/fixtures/mnist_sample.txt").unwrap().transform(|v: f64| v / 255.0);
    let y_train = utils::loader::matrix_from_txt("tests/fixtures/mnist_sample_labels.txt").unwrap().to_one_hot(10);

    let train_options = nn::TrainOptions::default().with_epochs(10).with_batch_size(5);
    network.fit(&x_train, &y_train, train_options);
}
