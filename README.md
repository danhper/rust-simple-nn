# simple_nn (work in progress)

Simple neural network implementation in Rust.

The project is still in progress, but the API looks like this.

```rust
use simple_nn::{nn, utils, OutputLayer};

let mut network = nn::Network::new();
network.add(nn::layers::Dense::new(784, 100));
network.add(nn::layers::Relu::new());
network.add(nn::layers::Dense::new(100, 100));
network.add(nn::layers::Relu::new());
network.add(nn::layers::Dense::new(100, 10));
network.add_final(nn::layers::Softmax::new().minimizing(nn::objectives::CrossEntropy::new()));
network.optimize_with(nn::optimizers::GradientDescent::new(0.5));

let X_train = utils::loader::matrix_from_txt("/path/to/train_data.txt").unwrap().transform(|v| v / 255.0)
let Y_train = utils::loader::matrix_from_txt("/path/to/train_labels.txt").unwrap().to_one_hot(10);

let X_test = utils::loader::matrix_from_txt("/path/to/test_data.txt").unwrap().transform(|v| v / 255.0);
let Y_test = utils::loader::matrix_from_txt("/path/to/test_labels.txt").unwrap().to_one_hot(10);

network.fit(X_train, Y_train, nn::TrainOptions::default());

let score = network.score(X_test, Y_test);

println!("Accuracy: {}", score);
```
