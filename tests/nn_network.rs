extern crate simple_nn;

use simple_nn::{layers, Network, Matrix};

#[test]
fn network_forward() {
    let input = Matrix::new_from(2, 3, vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]);
    let weights = Matrix::new_from(3, 3, vec![2.0, 4.0, 5.0, -8.0, 3.0, -1.0, 7.0, -2.0, 6.0]);

    let mut network = Network::new();
    network.add(layers::Dense::new_with_weights(&weights));
    network.add(layers::Relu::new());

    let results = network.forward(&input);
    assert_eq!(results[0], input);
    assert_eq!(results[1], Matrix::new_from(2, 3, vec![39.0, -8.0, 25.0, -74.0, 43.0, -21.0]));
    assert_eq!(results[2], Matrix::new_from(2, 3, vec![39.0, 0.0, 25.0, 0.0, 43.0, 0.0]));
}
