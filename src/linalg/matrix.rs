use std::{str, fmt, ops};
use rand;
use rand::distributions::{IndependentSample, Range};

#[derive(Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    elements: Vec<f64>
}

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

    pub fn one_hot(classes: usize, elements: Vec<usize>) -> Matrix {
        let mut matrix = Matrix::new(elements.len(), classes);
        for i in 0..elements.len() {
            matrix.set_at(i, elements[i], 1.0);
        }
        matrix
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

    pub fn to_one_hot(&self, classes: usize) -> Matrix {
        assert!(self.columns == 1, "matrix must be Nx1 to change to one_hot");
        let mut matrix = Matrix::new(self.elements.len(), classes);
        for i in 0..self.elements.len() {
            matrix.set_at(i, self.elements[i] as usize, 1.0);
        }
        matrix
    }

    pub fn at(&self, row: usize, column: usize) -> f64 {
        assert!(row < self.rows, "row is too large");
        assert!(column < self.columns, "column is too large");
        self.elements[row * self.columns + column]
    }

    pub fn set_at(&mut self, row: usize, column: usize, value: f64) {
        assert!(row < self.rows, "row is too large");
        assert!(column < self.columns, "column is too large");
        self.elements[row * self.columns + column] = value
    }

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    pub fn assert_same_size(&self, other: &Matrix) {
        assert!(self.rows == other.rows && self.columns == other.columns,
                "matrix should have same size, given {}x{} and {}x{}",
                self.rows, self.columns, other.rows, other.columns)
    }

    fn make_op<F>(&self, other: &Matrix, op: F) -> Matrix
            where F: FnMut(f64, f64) -> f64 {
        let mut output = Matrix::new_from(self.rows, self.columns, self.elements.to_owned());
        output.make_mut_op(other, op);
        output
    }

    fn make_mut_op<F>(&mut self, other: &Matrix, mut op: F)
            where F: FnMut(f64, f64) -> f64 {
        self.assert_same_size(other);
        for i in 0..self.size() {
            self.elements[i] = op(self.elements[i], other.elements[i]);
        }
    }

    pub fn add_mut(&mut self, other: &Matrix) {
        self.make_mut_op(other, |a, b| a + b)
    }

    pub fn sub_mut(&mut self, other: &Matrix) {
        self.make_mut_op(other, |a, b| a - b)
    }

    pub fn mul_mut(&mut self, other: &Matrix) {
        self.make_mut_op(other, |a, b| a * b)
    }

    pub fn div_mut(&mut self, other: &Matrix) {
        self.make_mut_op(other, |a, b| a / b)
    }

    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert!(self.columns == other.rows, "trying to multiply {}x{} with {}x{}",
            self.rows, self.columns, other.rows, other.columns);
        let mut output = Matrix::new(self.rows, other.columns);
        for row in 0..self.rows {
            for col in 0..other.columns {
                let mut v_ij = 0.0;
                for k in 0..self.columns {
                    v_ij += self.at(row, k) * other.at(k, col);
                }
                output.set_at(row, col, v_ij);
            }
        }
        output
    }

    pub fn t(&self) -> Matrix {
        let mut output = Matrix::new(self.columns, self.rows);
        for row in 0..self.rows {
            for col in 0..self.columns {
                output.set_at(col, row, self.at(row, col))
            }
        }
        output
    }

    pub fn clone(&self) -> Matrix {
        Matrix::new_from(self.rows, self.columns, self.elements.to_owned())
    }

    pub fn reduce_with_index<F, B>(&self, init: B, mut f: F) -> B
            where F: FnMut(B, f64, usize, usize) -> B {
        let mut result = init;
        for row in 0..self.rows {
            for col in 0..self.columns {
                result = f(result, self.at(row, col), row, col);
            }
        }
        result
    }

    pub fn reduce<F, B>(&self, init: B, mut f: F) -> B
            where F: FnMut(B, f64) -> B {
        self.reduce_with_index(init, |acc, v, _row, _col| f(acc, v))
    }

    pub fn reduce_rows_with_index<F>(&self, init: f64, mut f: F) -> Matrix
            where F: FnMut(f64, f64, usize, usize) -> f64 {
        let mut output = Matrix::new(self.rows, 1);
        for row in 0..self.rows {
            let mut result = init;
            for col in 0..self.columns {
                result = f(result, self.at(row, col), row, col);
            }
            output.set_at(row, 0, result);
        }
        output
    }

    pub fn reduce_rows<F>(&self, init: f64, mut f: F) -> Matrix
            where F: FnMut(f64, f64) -> f64 {
        self.reduce_rows_with_index(init, |acc, v, _row, _col| f(acc, v))
    }

    pub fn reduce_columns<F>(&self, init: f64, mut f: F) -> Matrix
            where F: FnMut(f64, f64) -> f64 {
        self.reduce_columns_with_index(init, |acc, v, _row, _col| f(acc, v))
    }

    pub fn reduce_columns_with_index<F>(&self, init: f64, mut f: F) -> Matrix
            where F: FnMut(f64, f64, usize, usize) -> f64 {
        let mut output = Matrix::new(1, self.columns);
        for col in 0..self.columns {
            let mut result = init;
            for row in 0..self.rows {
                result = f(result, self.at(row, col), row, col);
            }
            output.set_at(0, col, result);
        }
        output
    }

    pub fn transform<F>(&self, mut f: F) -> Matrix
            where F: FnMut(f64) -> f64 {
        self.transform_with_index(|v, _row, _col| f(v))
    }

    pub fn transform_with_index<F>(&self, mut f: F) -> Matrix
            where F: FnMut(f64, usize, usize) -> f64 {
        let mut output = Matrix::new(self.rows, self.columns);
        for row in 0..self.rows {
            for col in 0..self.columns {
                output.set_at(row, col, f(self.at(row, col), row, col));
            }
        }
        output
    }
}

impl<'a, 'b> ops::Add<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn add(self, other: &'b Matrix) -> Matrix {
        self.make_op(&other, |a, b| a + b)
    }
}

impl<'a, 'b> ops::Sub<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn sub(self, other: &'b Matrix) -> Matrix {
        self.make_op(&other, |a, b| a - b)
    }
}

impl<'a, 'b> ops::Mul<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn mul(self, other: &'b Matrix) -> Matrix {
        self.make_op(&other, |a, b| a * b)
    }
}

impl<'a, 'b> ops::Div<&'b Matrix> for &'a Matrix {
    type Output = Matrix;
    fn div(self, other: &'b Matrix) -> Matrix {
        self.make_op(&other, |a, b| a / b)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseMatrixError {
    message: String
}
impl ParseMatrixError {
    #[doc(hidden)]
    pub fn __description(&self) -> &str {
        "could not parse matrix"
    }
}

impl fmt::Display for ParseMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "could not parse matrix: {}", self.message)
    }
}

impl str::FromStr for Matrix {
    type Err = ParseMatrixError;
    fn from_str(s: &str) -> Result<Matrix, ParseMatrixError> {
        let lines: Vec<Vec<&str>> = s.trim().split('\n').map(|line| line.split(' ').collect()).collect();
        let mut matrix = Matrix::new(lines.len(), lines[0].len());
        for row in 0..matrix.rows {
            for col in 0..matrix.columns {
                match f64::from_str(lines[row][col]) {
                    Ok(v) => matrix.set_at(row, col, v),
                    Err(e) => return Err(ParseMatrixError { message: e.to_string() })
                }
            }
        }
        Ok(matrix)
    }
}
