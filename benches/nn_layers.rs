#![feature(test)]

extern crate simple_nn;
extern crate test;

use test::Bencher;

use simple_nn::{Matrix, layers};
use layers::Layer;


#[bench]
fn bench_layers_relu_compute(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(200, 200, -20.0, 20.0);
    let relu = layers::Relu::new();
    b.iter(|| relu.compute(&matrix));
}

#[bench]
fn bench_layers_dense_compute(b: &mut Bencher) {
    let matrix = Matrix::<f64>::random(200, 120, -10.0, 10.0);
    let weights = Matrix::<f64>::random(120, 80, -20.0, 20.0);
    let dense = layers::Dense::new_with_weights(&weights);
    b.iter(|| dense.compute(&matrix));
}
