use linalg::Matrix;
use nn::functions;
use nn::layers;

pub trait Objective<T: layers::OutputLayer> {
    fn loss(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64>;
    fn delta(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64>;
    fn predict_from_probs(&self, result: &Matrix<f64>) -> Matrix<u8>;
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

    fn predict_from_probs(&self, probs: &Matrix<f64>) -> Matrix<u8> {
        functions::argmax(&probs)
    }
}

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> BinaryCrossEntropy {
        BinaryCrossEntropy { }
    }
}

impl Objective<layers::Sigmoid> for BinaryCrossEntropy {
    fn loss(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64> {
        debug_assert_eq!(result.columns, 1, "binary cross entropy should have only one dimension");
        debug_assert_eq!(result.columns, 1, "binary cross entropy result should have only one dimension");
        result.reduce_rows_with_index(0.0, |_acc, v, row, col| {
            - (if expected.at(row, col) < 1e-5 { (1.0 - v).ln() } else { v.ln() })
        })
    }

    fn delta(&self, result: &Matrix<f64>, expected: &Matrix<f64>) -> Matrix<f64> {
        result.transform_with_index(|v, row, col| v - expected.at(row, col))
    }

    fn predict_from_probs(&self, probs: &Matrix<f64>) -> Matrix<u8> {
        probs.transform(|v| if v >= 0.5 { 1 } else { 0 } )
    }
}
