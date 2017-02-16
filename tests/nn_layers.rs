extern crate simple_nn;

use simple_nn::{Matrix, layers};
use layers::Layer;

#[test]
fn layers_relu_compute() {
    let matrix = Matrix::new_from(2, 3, vec![1.0, -1.0, 2.0, 4.0, -3.0, 0.5], true);
    let relu = layers::Relu::new();
    let expected = Matrix::new_from(2, 3, vec![1.0, 0.0, 2.0, 4.0, 0.0, 0.5], true);
    assert_eq!(relu.compute(&matrix), expected)
}

#[test]
fn layers_relu_delta() {
    let input_matrix = Matrix::new_from(2, 3, vec![-1.0, -1.0, 2.0, 4.0, 3.0, 0.5], true);
    let unused = Matrix::new(2, 3);
    let above_matrix = Matrix::new_from(2, 3, vec![1.0, -1.0, 2.0, 4.0, -3.0, 0.5], true);
    let relu = layers::Relu::new();
    let expected = Matrix::new_from(2, 3, vec![0.0, 0.0, 2.0, 4.0, -3.0, 0.5], true);
    assert_eq!(relu.delta(&input_matrix, &unused, &above_matrix), expected)
}

#[test]
fn layers_dense_compute() {
    let matrix = Matrix::new_from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let weights = Matrix::new_from(3, 2, vec![2.0, 4.0, 8.0, 3.0, 7.0, 2.0], true);
    let dense = layers::Dense::new_with_weights(&weights);
    let result = dense.compute(&matrix);
    let expected = Matrix::new_from(2, 2, vec![39.0, 16.0, 90.0, 43.0], true);
    assert_eq!(result, expected);
}

#[test]
fn layers_dense_delta() {
    let above = Matrix::new_from(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
    let weights = Matrix::new_from(3, 2, vec![2.0, 4.0, 8.0, 3.0, 7.0, 2.0], true);
    let dense = layers::Dense::new_with_weights(&weights);
    let result = dense.delta(&Matrix::new(2, 2), &Matrix::new(2, 2), &above);
    let expected = Matrix::new_from(4, 3, vec![10.0, 14.0, 11.0, 22.0, 36.0, 29.0, 34.0, 58.0, 47.0, 46.0, 80.0, 65.0], true);
    assert_eq!(result, expected);
}

#[test]
fn layers_softmax_compute() {
    let input = Matrix::new_from(1, 7, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], true);
    let softmax = layers::Softmax::new();
    let output = softmax.compute(&input);
    assert_eq!(output.rows, 1);
    assert_eq!(output.columns, 7);
    assert!((output.at(0, 0) - 0.02364054).abs() < 1e-5);
}

#[test]
fn layers_softmax_delta() {
    let outgoing = Matrix::new_from(1, 7, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0], true);
    let above = Matrix::new_from(1, 7, vec![4.0, 2.0, 2.0, 4.0, 1.0, 2.0, 3.0], true);
    let softmax = layers::Softmax::new();
    let result = softmax.delta(&Matrix::new(2, 2), &outgoing, &above);
    let expected = Matrix::new_from(1, 7, vec![-40.0, -84.0, -126.0, -160.0, -43.0, -84.0, -123.0], true);
    assert_eq!(result, expected);
}
