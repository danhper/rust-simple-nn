# simple_nn
[![Build Status](https://travis-ci.org/tuvistavie/rust-simple-nn.svg?branch=master)](https://travis-ci.org/tuvistavie/rust-simple-nn)

Simple neural network implementation in Rust.

NOTE: I wanted to give Rust a try, and decided to try implementing a simple NN framework,
but this is not meant to be used in production (it is way too slow for now anyway).

Here is a small example for the mnist dataset.

```rust
use simple_nn::{nn, utils};

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

let x_train = utils::loader::matrix_from_txt("/path/to/train_data.txt").unwrap().transform(|v| v / 255.0)
let y_train = utils::loader::matrix_from_txt("/path/to/train_labels.txt").unwrap().to_one_hot(10);

let x_test = utils::loader::matrix_from_txt("/path/to/test_data.txt").unwrap().transform(|v| v / 255.0);
let y_test = utils::loader::matrix_from_txt("/path/to/test_labels.txt").unwrap().to_one_hot(10);

let train_options = nn::TrainOptions::default().with_epochs(10).with_batch_size(256);
network.fit(&x_train, &y_train, train_options);

let loss = network.loss(&x_test, &y_test);

println!("loss: {}", loss);
```

## Progress

Only very few functions have been implemented yet.
Help is very welcome.

### Layers

- [x] Linear (missing bias)
- [x] ReLU
- [ ] sigmoid
- [ ] tanh
- [ ] softplus
- [ ] softsign

### Objectives

- [x] Categorical Cross Entropy
- [ ] Mean square
- [ ] Poisson
- [ ] KL divergence

### Optimizers

- [x] SGD
- [ ] Adam
- [ ] Adamax
- [ ] RMSprop

### Other

- [ ] Serialization
- [ ] Metrics
- [ ] Layer configurations
