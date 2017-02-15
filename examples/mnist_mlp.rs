extern crate simple_nn;

use simple_nn::{nn, utils};

fn main() {
    let mut network = nn::NetworkBuilder::new()
        .add(nn::layers::Dense::new(784, 100))
        .add(nn::layers::Relu::new())
        .add(nn::layers::Dense::new(100, 100))
        .add(nn::layers::Relu::new())
        .add(nn::layers::Dense::new(100, 10))
        .add_output(nn::layers::Softmax::new())
        .minimize(nn::objectives::CrossEntropy::new())
        .with(nn::optimizers::SGD::new(0.1))
        .build();


    let x_train = utils::loader::matrix_from_txt("tests/fixtures/mnist_sample.txt").unwrap().transform(|v: f64| v / 255.0);
    let y_train = utils::loader::matrix_from_txt("tests/fixtures/mnist_sample_labels.txt").unwrap().to_one_hot(10);

    println!("start training...");
    let train_options = nn::TrainOptions::default().with_epochs(10).with_batch_size(256);
    network.fit(&x_train, &y_train, train_options);
}
