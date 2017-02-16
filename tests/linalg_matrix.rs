extern crate simple_nn;

use std::str::FromStr;
use simple_nn::{Matrix};

#[test]
fn matrix_creation() {
    let matrix = Matrix::<f64>::new(10, 5);
    assert_eq!(matrix.rows, 10);
    assert_eq!(matrix.columns, 5);
}

#[test]
fn matrix_creation_from_vector() {
    let matrix = simple_nn::Matrix::new_from(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], true);
    assert_eq!(matrix.rows, 2);
    assert_eq!(matrix.columns, 3);
    assert_eq!(matrix.at(1, 1), 0.5);
}

#[test]
fn matrix_add() {
    let matrix = Matrix::new_from(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], true);
    let other = Matrix::new_from(2, 3, vec![1.0, 2.0, -1.0, 3.0, -2.0, 0.0], true);
    let result = &matrix + &other;
    let expected = Matrix::new_from(2, 3, vec![1.1, 2.2, -0.7, 3.4, -1.5, 0.6], true);
    assert_eq!(expected, result);
}

#[test]
#[should_panic]
fn matrix_add_bad_size() {
    let matrix = Matrix::new_from(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], true);
    let other = Matrix::new_from(2, 2, vec![1.0, 2.0, -1.0, 3.0], true);
    &matrix + &other;
}

#[test]
fn matrix_sub() {
    let matrix = Matrix::new_from(2, 3, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], true);
    let other = Matrix::new_from(2, 3, vec![1.0, 2.0, -1.0, 3.0, -2.0, 0.0], true);
    let result = &matrix - &other;
    let expected = Matrix::new_from(2, 3, vec![-0.9, -1.8, 1.3, -2.6, 2.5, 0.6], true);
    assert_eq!(result, expected);
}

#[test]
fn matrix_mul() {
    let matrix = Matrix::new_from(2, 3, vec![8.0, 9.0, 12.0, 15.0, 21.0, 42.0], true);
    let other = Matrix::new_from(2, 3, vec![2.0, 3.0, 2.0, 3.0, 3.0, 7.0], true);
    let result = &matrix * &other;
    let expected = Matrix::new_from(2, 3, vec![16.0, 27.0, 24.0, 45.0, 63.0, 294.0], true);
    assert_eq!(result, expected);
}

#[test]
fn matrix_div() {
    let matrix = Matrix::new_from(2, 3, vec![8.0, 9.0, 12.0, 15.0, 21.0, 42.0], true);
    let other = Matrix::new_from(2, 3, vec![2.0, 3.0, 2.0, 3.0, 3.0, 7.0], true);
    let result = &matrix / &other;
    let expected = Matrix::new_from(2, 3, vec![4.0, 3.0, 6.0, 5.0, 7.0, 6.0], true);
    assert_eq!(result, expected);
}

#[test]
fn matrix_matmul() {
    let matrix = Matrix::new_from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let other = Matrix::new_from(3, 2, vec![2.0, 4.0, 8.0, 3.0, 7.0, 2.0], true);
    let result = matrix.matmul(&other);
    let expected = Matrix::new_from(2, 2, vec![39.0, 16.0, 90.0, 43.0], true);
    assert_eq!(result, expected);
}

#[test]
fn matrix_t() {
    let matrix = Matrix::new_from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let result = matrix.t();
    let expected = Matrix::new_from(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
    assert_eq!(result, expected);
    assert_eq!(matrix, result.t());
}

#[test]
fn matrix_from_str() {
    let string = "1.1 2 3 4\n5 6 7.70e+01 8\n9 10 11 12";
    let expected = Matrix::new_from(3, 4, vec![1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.70e+01, 8.0, 9.0, 10.0, 11.0, 12.0], true);
    assert_eq!(Matrix::from_str(string).unwrap(), expected);
}

#[test]
fn slice_rows() {
    let matrix = Matrix::new_from(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], true);
    let sliced = matrix.slice_rows(1..3);
    assert_eq!(sliced, Matrix::new_from(2, 2, vec![3.0, 4.0, 5.0, 6.0], true));
}

#[test]
fn strassen_mul() {
    let matrix = Matrix::<u64>::random(100, 100, -20, 20);
    let other = Matrix::<u64>::random(100, 100, -20, 20);
    assert_eq!(matrix.matmul(&other), matrix.serial_matmul(&other));
}
