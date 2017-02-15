use linalg::Matrix;
use nn::functions;
use nn::layers;

pub trait Objective<T: layers::OutputLayer> {
    fn loss(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64>;
    fn delta(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64>;
}

pub struct CrossEntropy;

impl CrossEntropy {
    pub fn new() -> CrossEntropy {
        CrossEntropy { }
    }
}

impl Objective<layers::Softmax> for CrossEntropy {
    fn loss(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64> {
        functions::cross_entropy_from_probs(result, expected)
    }

    fn delta(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64> {
        result.transform_with_index(|v, row, col| v - expected.at(row, col))
    }
}
