#![feature(test)]

extern crate simple_nn;
extern crate test;

use test::Bencher;

use simple_nn::{Matrix};


#[bench]
fn bench_matrix_add(b: &mut Bencher) {
    let matrix = Matrix::random(200, 200, -10.0, 10.0);
    let other = Matrix::random(200, 200, -20.0, 20.0);
    b.iter(|| matrix.add(&other));
}

#[bench]
fn bench_matrix_sub(b: &mut Bencher) {
    let matrix = Matrix::random(200, 200, -10.0, 10.0);
    let other = Matrix::random(200, 200, -20.0, 20.0);
    b.iter(|| matrix.sub(&other));
}

#[bench]
fn bench_matrix_matmul(b: &mut Bencher) {
    let matrix = Matrix::random(200, 120, -10.0, 10.0);
    let other = Matrix::random(120, 80, -20.0, 20.0);
    b.iter(|| matrix.matmul(&other));
}
