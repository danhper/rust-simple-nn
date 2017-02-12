use std;
use rand;
use rand::distributions::{IndependentSample, Range};

#[derive(Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    elements: Vec<f64>
}

#[allow(dead_code)]
impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        let mut elems = Vec::with_capacity(rows * columns);
        elems.resize(rows * columns, 0.0);
        Matrix {
            rows: rows,
            columns: columns,
            elements: elems,
        }
    }

    pub fn new_from(rows: usize, columns: usize, elements: Vec<f64>) -> Matrix {
        assert!(rows * columns == elements.len());
        Matrix {
            rows: rows,
            columns: columns,
            elements: elements,
        }
    }

    pub fn random(rows: usize, columns: usize, min: f64, max: f64) -> Matrix {
        let between = Range::new(min, max);
        let mut rng = rand::thread_rng();

        let mut matrix = Matrix::new(rows, columns);
        for i in 0..(rows * columns) {
            matrix.elements[i] = between.ind_sample(&mut rng);
        }
        matrix
    }

    pub fn at(&self, row: usize, column: usize) -> f64 {
        // assert!(row < self.rows, "row is too large");
        // assert!(column < self.columns, "column is too large");
        self.elements[row * self.columns + column]
    }

    pub fn set_at(&mut self, row: usize, column: usize, value: f64) {
        // assert!(row < self.rows, "row is too large");
        // assert!(column < self.columns, "column is too large");
        self.elements[row * self.columns + column] = value
    }

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    fn assert_same_size(&self, other: &Matrix) {
        assert!(self.rows == other.rows && self.columns == other.columns, "matrix should have same size")
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut output = Matrix::new_from(self.rows, self.columns, self.elements.to_owned());
        output.add_mut(other);
        output
    }

    pub fn add_mut(&mut self, other: &Matrix) {
        self.assert_same_size(other);
        for i in 0..self.size() {
            self.elements[i] += other.elements[i]
        }
    }

    pub fn sub(&self, other: &Matrix) -> Matrix {
        let mut output = Matrix::new_from(self.rows, self.columns, self.elements.to_owned());
        output.sub_mut(other);
        output
    }

    pub fn sub_mut(&mut self, other: &Matrix) {
        self.assert_same_size(other);
        for i in 0..self.size() {
            self.elements[i] -= other.elements[i]
        }
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert!(self.columns == other.rows);
        let mut output = Matrix::new(self.rows, other.columns);
        for i in 0..self.rows {
            for j in 0..other.columns {
                let mut v_ij = 0.0;
                for k in 0..self.columns {
                    v_ij += self.at(i, k) * other.at(k, j);
                }
                output.set_at(i, j, v_ij);
            }
        }
        output
    }

    pub fn t(&self) -> Matrix {
        let mut output = Matrix::new(self.columns, self.rows);
        for i in 0..self.rows {
            for j in 0..self.columns {
                output.set_at(j, i, self.at(i, j))
            }
        }
        output
    }

    pub fn clone(&self) -> Matrix {
        Matrix::new_from(self.rows, self.columns, self.elements.to_owned())
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut elements = String::new();
        for row in 0..self.rows {
            for column in 0..self.columns {
                elements.push_str(&format!("{} ", self.at(row, column).to_string()));
            }
            elements.push_str("\n");
        }

        let separator = "--------------";
        let formatted = format!("Matrix {}x{}\n{}\n{}{}", self.rows, self.columns, separator, elements, separator);

        write!(fmt, "{}", formatted)
    }
}
