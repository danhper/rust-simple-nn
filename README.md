# simple_nn (work in progress)

Simple neural network implementation in Rust.

The project is still in progress, but the interface will look like this.

```rust
use simple_nn::{layers, Network, Matrix, Trainer};

let mut network = Network::new();
network.add(layers::Dense::new(784, 100));
network.add(layers::Relu::new());
network.add(layers::Dense::new(100, 100));
network.add(layers::Relu::new());
network.add(layers::Dense::new(100, 10));
network.add(layers::Softmax::new());
```
