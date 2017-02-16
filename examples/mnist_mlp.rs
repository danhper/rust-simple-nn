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
        .with(nn::optimizers::SGD::new(0.5))
        .build();

    println!("loading training data...");

    let x_train = utils::loader::matrix_from_txt("data/train_x_60000x784_float32.txt").unwrap().transform(|v: f64| v / 255.0);
    let y_train = utils::loader::matrix_from_txt("data/train_y_60000_int32.txt").unwrap().to_one_hot(10);

    println!("start training...");

    let train_options = nn::TrainOptions::default().with_epochs(3).with_batch_size(256);
    network.fit(&x_train, &y_train, train_options);

    println!("training done");

    println!("loading test data...");

    let x_test = utils::loader::matrix_from_txt("data/test_x_10000x784_float32.txt").unwrap().transform(|v: f64| v / 255.0);
    let y_test = utils::loader::matrix_from_txt("data/test_y_10000_int32.txt").unwrap().to_one_hot(10);

    let predict_probs = network.predict_probs(&x_test);
    let loss = network.mean_loss_from_probs(&predict_probs, &y_test);
    let accuracy = network.accuracy_from_probs(&predict_probs, &y_test);
    println!("accuracy = {}, mean loss = {}", accuracy, loss);
}
