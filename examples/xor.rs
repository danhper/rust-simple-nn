extern crate simple_nn;
extern crate rand;

use rand::distributions::{IndependentSample, Range};

use simple_nn::{nn, Matrix};


fn generate_xor_data(n: usize) -> (Matrix<f64>, Matrix<f64>) {
    let mut x_v = Matrix::new(n, 2);
    let mut y_v = Matrix::new(n, 1);
    let between = Range::new(0, 2);
    let mut rng = rand::thread_rng();
    for i in 0..n {
        let x = between.ind_sample(&mut rng);
        let y = between.ind_sample(&mut rng);
        x_v.set_at(i, 0, x as f64);
        x_v.set_at(i, 1, y as f64);
        y_v.set_at(i, 0, (x ^ y) as f64);
    }
    (x_v, y_v)
}

fn main() {
    let mut network = nn::NetworkBuilder::new()
        .add(nn::layers::Dense::new(2, 10))
        .add(nn::layers::Sigmoid::new())
        .add(nn::layers::Dense::new(10, 1))
        .add_output(nn::layers::Sigmoid::new())
        .minimize(nn::objectives::BinaryCrossEntropy::new())
        .with(nn::optimizers::SGD::new(0.1))
        .build();

    let (x_train, y_train) = generate_xor_data(1_000_000);
    let train_options = nn::TrainOptions::default().with_epochs(3).with_batch_size(64);
    network.fit(&x_train, &y_train, train_options);

    let x_test = Matrix::new_from(4, 2, vec![0, 0, 0, 1, 1, 0, 1, 1], true);
    let results = network.predict(&x_test);
    for i in 0..results.rows {
        println!("{} ^ {} = {}", x_test.at(i, 0), x_test.at(i, 1), results.at(i, 0));
    }
}
