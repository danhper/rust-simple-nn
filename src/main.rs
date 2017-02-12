extern crate rand;

mod linalg;
mod nn;

fn main() {
    let matrix = linalg::Matrix::random(10, 5, -1.0, 1.0);
    println!("{}", matrix);
}
