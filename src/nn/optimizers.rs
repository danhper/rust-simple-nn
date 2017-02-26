use linalg::{Matrix};

pub trait Optimizer {
    fn apply_gradients(&self, weights: &mut Matrix<f64>, gradients: &Matrix<f64>);
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
    fn apply_gradients(&self, weights: &mut Matrix<f64>, gradients: &Matrix<f64>) {
        weights.sub_mut(&gradients.transform(|v| v * self.learning_rate));
    }
}
