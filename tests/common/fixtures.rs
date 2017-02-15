use std::env;
use std::path::Path;
use std::convert::AsRef;

use simple_nn::utils::loader;
use simple_nn::Matrix;

pub fn load_matrix<S: AsRef<Path>>(name: S) -> Matrix {
    let root = env::current_dir().unwrap();
    let filepath = Path::new(root.as_path()).join(String::from("tests/fixtures")).join(name);
    loader::matrix_from_txt(filepath).unwrap()
}
