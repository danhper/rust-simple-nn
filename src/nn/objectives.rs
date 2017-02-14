use linalg::Matrix;
use nn::functions;
use nn::layers;

pub trait Objective {
    fn loss(&self, result: &Matrix, expected: &Matrix) -> Matrix;
    fn delta(&self, result: &Matrix, expected: &Matrix) -> Matrix;
}

pub struct CrossEntropy {
    activation: Option<Box<CrossEntropyComputation>>
}

impl CrossEntropy {
    pub fn new() -> CrossEntropy {
        CrossEntropy { activation: None }
    }

    pub fn set_activation(&mut self, activation: Box<CrossEntropyComputation>) {
        self.activation = Some(activation)
    }
}

pub trait CrossEntropyComputation {
    fn delta(&self, result: &Matrix, expected: &Matrix) -> Matrix;
}

impl CrossEntropyComputation for layers::Softmax {
    fn delta(&self, result: &Matrix, expected: &Matrix) -> Matrix {
        result.transform_with_index(|v, row, col| v - expected.at(row, col))
    }
}

impl Objective for CrossEntropy {
    fn loss(&self, result: &Matrix, expected: &Matrix) -> Matrix {
        functions::cross_entropy_from_probs(result, expected)
    }

    fn delta(&self, result: &Matrix, expected: &Matrix) -> Matrix {
        self.activation.as_ref().unwrap().delta(result, expected)
    }
}
