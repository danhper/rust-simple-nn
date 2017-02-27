use std::{env, str, fmt};
use std::path::Path;
use std::convert::AsRef;

use rand;
use rand::distributions::{IndependentSample, Range};

use simple_nn::utils::loader;
use simple_nn::Matrix;

pub fn load_matrix<T, S: AsRef<Path>>(name: S) -> Matrix<T>
        where T: str::FromStr, <T as str::FromStr>::Err: fmt::Display {
    let root = env::current_dir().unwrap();
    let filepath = Path::new(root.as_path()).join(String::from("tests/fixtures")).join(name);
    loader::matrix_from_txt(filepath).unwrap()
}

#[allow(dead_code)]
pub fn generate_xor_data(n: usize) -> (Matrix<f64>, Matrix<f64>) {
    let mut x_v = Matrix::new(n, 2);
    let mut y_v = Matrix::new(n, 1);
    let between = Range::new(0, 2);
    let mut rng = rand::thread_rng();
    for i in 0..n {
        let x = between.ind_sample(&mut rng);
        let y = between.ind_sample(&mut rng);
        x_v.set_at(i, 0, x as f64);
        x_v.set_at(i, 1, y as f64);
        y_v.set_at(i, 0, (x ^ y) as f64);
    }
    (x_v, y_v)
}
