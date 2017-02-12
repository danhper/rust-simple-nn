extern crate simple_nn;

use simple_nn::{Matrix, layers};
use layers::Layer;

#[test]
fn layers_relu_compute() {
    let matrix = Matrix::new_from(2, 3, vec![1.0, -1.0, 2.0, 4.0, -3.0, 0.5]);
    let relu = layers::Relu::new();
    let expected = Matrix::new_from(2, 3, vec![1.0, 0.0, 2.0, 4.0, 0.0, 0.5]);
    assert_eq!(relu.compute(&matrix), expected)
}

#[test]
fn layers_relu_delta() {
    let input_matrix = Matrix::new_from(2, 3, vec![-1.0, -1.0, 2.0, 4.0, 3.0, 0.5]);
    let unused = Matrix::new(2, 3);
    let above_matrix = Matrix::new_from(2, 3, vec![1.0, -1.0, 2.0, 4.0, -3.0, 0.5]);
    let relu = layers::Relu::new();
    let expected = Matrix::new_from(2, 3, vec![0.0, 0.0, 2.0, 4.0, -3.0, 0.5]);
    assert_eq!(relu.delta(&input_matrix, &unused, &above_matrix), expected)
}

#[test]
fn layers_dense_compute() {
    let matrix = Matrix::new_from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let weights = Matrix::new_from(3, 2, vec![2.0, 4.0, 8.0, 3.0, 7.0, 2.0]);
    let dense = layers::Dense::new_with_weights(&weights);
    let result = dense.compute(&matrix);
    let expected = Matrix::new_from(2, 2, vec![39.0, 16.0, 90.0, 43.0]);
    assert_eq!(result, expected);
}
