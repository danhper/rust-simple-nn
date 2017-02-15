extern crate rand;
extern crate simple_nn;

mod common;
use common::fixtures;

use simple_nn::Matrix;

#[test]
fn utils_load_matrix_from_txt() {
    let matrix: Matrix<f64> = fixtures::load_matrix("mnist_sample.txt");
    assert_eq!(matrix.rows, 10);
    assert_eq!(matrix.columns, 784);

    let matrix: Matrix<usize> = fixtures::load_matrix("mnist_sample_labels.txt");
    assert_eq!(matrix.rows, 10);
    assert_eq!(matrix.columns, 1);
}
