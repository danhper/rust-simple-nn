use linalg::{Matrix};

pub trait Optimizer {
    fn apply_gradients(&self, weights: &mut Matrix<f64>, gradients: &Matrix<f64>, inputs_count: usize);
}

#[derive(Clone)]
pub struct SGD {
    pub learning_rate: f64
}

impl SGD {
    pub fn new(learning_rate: f64) -> SGD {
        SGD { learning_rate: learning_rate }
    }
}

impl Optimizer for SGD {
    fn apply_gradients(&self, weights: &mut Matrix<f64>, gradients: &Matrix<f64>, inputs_count: usize) {
        weights.sub_mut(&gradients.transform(|v| v / (inputs_count as f64) * self.learning_rate));
    }
}
