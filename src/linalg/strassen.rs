use linalg::matrix::Matrix;
use std::{ops, thread, marker};

const MIN_SIZE: usize = 64;
const THREAD_MAX_DEPTH: u8 = 1;

pub fn mul<T>(matrix: &Matrix<T>, other: &Matrix<T>) -> Matrix<T>
        where T: Default + Clone + ops::Add<Output = T> + ops::Sub<Output = T> + ops::Mul<Output = T> + marker::Send + 'static {
    strassen_mul(matrix, other, 1)
}

fn strassen_mul<T>(matrix: &Matrix<T>, other: &Matrix<T>, depth: u8) -> Matrix<T>
        where T: Default + Clone + ops::Add<Output = T> + ops::Sub<Output = T> + ops::Mul<Output = T> + marker::Send + 'static {

    if matrix.rows < MIN_SIZE || matrix.rows % 2 != 0 {
        return matrix.serial_matmul(other);
    }

    let rows = matrix.rows / 2;
    let columns = matrix.columns / 2;

    let mut a11 = Matrix::new(rows, columns);
    let mut a12 = Matrix::new(rows, columns);
    let mut a21 = Matrix::new(rows, columns);
    let mut a22 = Matrix::new(rows, columns);

    let mut b11 = Matrix::new(rows, columns);
    let mut b12 = Matrix::new(rows, columns);
    let mut b21 = Matrix::new(rows, columns);
    let mut b22 = Matrix::new(rows, columns);

    for row in 0..rows {
        for col in 0..columns {
            a11.set_at(row, col, matrix.at(row, col));
            a12.set_at(row, col, matrix.at(row, col + columns));
            a21.set_at(row, col, matrix.at(row + rows, col));
            a22.set_at(row, col, matrix.at(row + rows, col + columns));

            b11.set_at(row, col, other.at(row, col));
            b12.set_at(row, col, other.at(row, col + columns));
            b21.set_at(row, col, other.at(row + rows, col));
            b22.set_at(row, col, other.at(row + rows, col + columns));
        }
    }

    let (p1, p2, p3, p4, p5, p6, p7) = if depth <= THREAD_MAX_DEPTH {
        mul_multi_thread(&a11, &a12, &a21, &a22, &b11, &b12, &b21, &b22, depth)
    } else {
        mul_single_thread(&a11, &a12, &a21, &a22, &b11, &b12, &b21, &b22, depth)
    };

    let c11 = &(&(&p1 + &p4) - &p5) + &p7;
    let c12 = &p3 + &p5;
    let c21 = &p2 + &p4;
    let c22 = &(&(&p1 - &p2) + &p3) + &p6;

    let mut output = Matrix::new(matrix.rows, other.columns);
    for row in 0..rows {
        for col in 0..columns {
            output.set_at(row, col, c11.at(row, col));
            output.set_at(row, col + columns, c12.at(row, col));
            output.set_at(row + rows, col, c21.at(row, col));
            output.set_at(row + rows, col + columns, c22.at(row, col));
        }
    }
    output
}

fn mul_single_thread<T>(a11: &Matrix<T>, a12: &Matrix<T>, a21: &Matrix<T>, a22: &Matrix<T>,
                     b11: &Matrix<T>, b12: &Matrix<T>, b21: &Matrix<T>, b22: &Matrix<T>, depth: u8) ->
                        (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>)
        where T: Default + Clone + ops::Add<Output = T> + ops::Sub<Output = T> + ops::Mul<Output = T> + marker::Send + 'static {
    let p1 = strassen_mul(&(a11 + a22), &(b11 + b22), depth + 1);
    let p2 = strassen_mul(&(a21 + a22), b11, depth + 1);
    let p3 = strassen_mul(a11, &(b12 - b22), depth + 1);
    let p4 = strassen_mul(&a22, &(b21 - b11), depth + 1);
    let p5 = strassen_mul(&(a11 + a12), b22, depth + 1);
    let p6 = strassen_mul(&(a21 - a11), &(b11 + b12), depth + 1);
    let p7 = strassen_mul(&(a12 - a22), &(b21 + b22), depth + 1);
    (p1, p2, p3, p4, p5, p6, p7)
}

fn mul_multi_thread<T>(a11: &Matrix<T>, a12: &Matrix<T>, a21: &Matrix<T>, a22: &Matrix<T>,
                     b11: &Matrix<T>, b12: &Matrix<T>, b21: &Matrix<T>, b22: &Matrix<T>, depth: u8) ->
                        (Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>, Matrix<T>)
        where T: Default + Clone + ops::Add<Output = T> + ops::Sub<Output = T> + ops::Mul<Output = T> + marker::Send + 'static {


    let (a11_c1, a22_c1, b11_c1, b22_c1) = (a11.clone(), a22.clone(), b11.clone(), b22.clone());
    let child1 = thread::spawn(move || strassen_mul(&(&a11_c1 + &a22_c1), &(&b11_c1 + &b22_c1), depth + 1));

    let (a21_c2, a22_c2, b11_c2) = (a21.clone(), a22.clone(), b11.clone());
    let child2 = thread::spawn(move || strassen_mul(&(&a21_c2 + &a22_c2), &b11_c2, depth + 1));

    let (a11_c3, b12_c3, b22_c3) = (a11.clone(), b12.clone(), b22.clone());
    let child3 = thread::spawn(move || strassen_mul(&a11_c3, &(&b12_c3 - &b22_c3), depth + 1));

    let (a22_c4, b21_c4, b11_c4) = (a22.clone(), b21.clone(), b11.clone());
    let child4 = thread::spawn(move || strassen_mul(&a22_c4, &(&b21_c4 - &b11_c4), depth + 1));

    let (a11_c5, a12_c5, b22_c5) = (a11.clone(), a12.clone(), b22.clone());
    let child5 = thread::spawn(move || strassen_mul(&(&a11_c5 + &a12_c5), &b22_c5, depth + 1));

    let (a11_c6, a21_c6, b11_c6, b12_c6) = (a11.clone(), a21.clone(), b11.clone(), b12.clone());
    let child6 = thread::spawn(move || strassen_mul(&(&a21_c6 - &a11_c6), &(&b11_c6 + &b12_c6), depth + 1));

    let (a12_c7, a22_c7, b21_c7, b22_c7) = (a12.clone(), a22.clone(), b21.clone(), b22.clone());
    let child7 = thread::spawn(move || strassen_mul(&(&a12_c7 - &a22_c7), &(&b21_c7 + &b22_c7), depth + 1));

    (child1.join().unwrap(), child2.join().unwrap(), child3.join().unwrap(),
     child4.join().unwrap(), child5.join().unwrap(), child6.join().unwrap(), child7.join().unwrap())
}
