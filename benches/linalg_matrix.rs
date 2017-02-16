#![feature(test)]

extern crate simple_nn;
extern crate test;

use test::Bencher;

use simple_nn::Matrix;


#[bench]
fn bench_matrix_add(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(200, 200, -10.0, 10.0);
    let other = Matrix::<f64>::random(200, 200, -20.0, 20.0);
    b.iter(|| &matrix + &other);
}

#[bench]
fn bench_matrix_sub(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(200, 200, -10.0, 10.0);
    let other = Matrix::<f64>::random(200, 200, -20.0, 20.0);
    b.iter(|| &matrix - &other);
}

#[bench]
fn bench_matrix_serial_matmul(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(300, 300, -10.0, 10.0);
    let other = Matrix::<f64>::random(300, 300, -20.0, 20.0);
    b.iter(|| matrix.serial_matmul(&other));
}

#[bench]
fn bench_matrix_matmul(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(300, 300, -10.0, 10.0);
    let other = Matrix::<f64>::random(300, 300, -20.0, 20.0);
    b.iter(|| matrix.matmul(&other));
}

#[bench]
fn bench_matrix_t(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(200, 120, -10.0, 10.0);
    b.iter(|| matrix.t());
}
