use linalg::{Matrix};

pub trait Optimizer {
    fn apply_gradients(&self, weights: &mut Matrix<f64>, gradients: &Matrix<f64>);
    fn boxed(&self) -> Box<Optimizer>;
}

#[derive(Clone)]
pub struct SGD {
    pub learning_rate: f64
}

impl SGD {
    pub fn new(learning_rate: f64) -> Box<SGD> {
        Box::new(SGD { learning_rate: learning_rate })
    }
}

impl Optimizer for SGD {
    fn apply_gradients(&self, weights: &mut Matrix<f64>, gradients: &Matrix<f64>) {
        weights.sub_mut(&gradients.transform(|v| v * self.learning_rate));
    }
    fn boxed(&self) -> Box<Optimizer> {
        Box::new(self.clone())
    }
}
