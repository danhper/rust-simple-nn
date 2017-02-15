# simple_nn (work in progress)

Simple neural network implementation in Rust.

The project is still in progress, but the API looks like this.

```rust
use simple_nn::{nn, utils, OutputLayer};

let mut network = nn::NetworkBuilder::new()
    .add(nn::layers::Dense::new(784, 100))
    .add(nn::layers::Relu::new())
    .add(nn::layers::Dense::new(100, 100))
    .add(nn::layers::Relu::new())
    .add(nn::layers::Dense::new(100, 10))
    .add_output(nn::layers::Softmax::new())
    .minimize(nn::objectives::CrossEntropy::new())
    .with(nn::optimizers::SGD::new(0.01))
    .build();

let x_train = utils::loader::matrix_from_txt("/path/to/train_data.txt").unwrap().transform(|v| v / 255.0)
let y_train = utils::loader::matrix_from_txt("/path/to/train_labels.txt").unwrap().to_one_hot(10);

let x_test = utils::loader::matrix_from_txt("/path/to/test_data.txt").unwrap().transform(|v| v / 255.0);
let y_test = utils::loader::matrix_from_txt("/path/to/test_labels.txt").unwrap().to_one_hot(10);


let train_options = nn::TrainOptions::default().with_epochs(10).with_batch_size(5);
network.fit(&x_train, &y_train, train_options);

let loss = network.loss(x_test, y_test);

println!("loss: {}", loss);
```
