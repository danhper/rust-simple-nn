use linalg::Matrix;
use nn::functions;

pub trait Layer {
    fn compute(&self, incoming: &Matrix) -> Matrix;
    fn delta(&self, incoming: &Matrix, outgoing: &Matrix, above: &Matrix) -> Matrix;
    fn has_trainable_weights(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct Relu {
    threshold: f64
}

#[allow(dead_code)]
impl Relu {
    pub fn new() -> Box<Relu> {
        Box::new(Relu { threshold: 0.0 })
    }

    fn compute_in_out(&self, input: &Matrix, above: &Matrix) -> Matrix {
        let mut output = Matrix::new(input.rows, input.columns);
        for i in 0..input.rows {
            for j in 0..input.columns {
                let value = if input.at(i, j) > self.threshold { above.at(i, j) } else { self.threshold };
                output.set_at(i, j, value);
            }
        }
        output
    }
}

impl Layer for Relu {
    fn compute(&self, incoming: &Matrix) -> Matrix {
        self.compute_in_out(incoming, incoming)
    }

    #[allow(unused_variables)]
    fn delta(&self, incoming: &Matrix, outgoing: &Matrix, above: &Matrix) -> Matrix {
        self.compute_in_out(incoming, above)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct Dense {
    weights: Matrix,
    pub input_dim: usize,
    pub output_dim: usize
}

#[allow(dead_code)]
impl Dense {
    pub fn new(output_dim: usize, input_dim: usize) -> Box<Dense> {
        let matrix = Matrix::random(input_dim, output_dim, -1.0, 1.0);
        Box::new(Dense {
            weights: matrix,
            input_dim: input_dim,
            output_dim: output_dim
        })
    }

    pub fn new_with_weights(weights: &Matrix) -> Box<Dense> {
        Box::new(Dense {
            weights: weights.clone(),
            input_dim: weights.rows.to_owned(),
            output_dim: weights.columns.to_owned()
        })
    }
}

impl Layer for Dense {
    fn has_trainable_weights(&self) -> bool {
        true
    }

    fn compute(&self, incoming: &Matrix) -> Matrix {
        incoming.matmul(&self.weights)
    }

    #[allow(unused_variables)]
    fn delta(&self, incoming: &Matrix, outgoing: &Matrix, above: &Matrix) -> Matrix {
        above.matmul(&self.weights.t())
    }
}

#[allow(dead_code)]
struct Softmax;

#[allow(dead_code)]
impl Layer for Softmax {
    fn compute(&self, incoming: &Matrix) -> Matrix {
        return functions::softmax(incoming);
    }

    #[allow(unused_variables)]
    fn delta(&self, incoming: &Matrix, outgoing: &Matrix, above: &Matrix) -> Matrix {
        return functions::softmax(incoming);
    }
}
