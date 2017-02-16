use std::{str, fmt, ops};
use rand;
use rand::distributions::{IndependentSample, Range};

#[derive(Debug, PartialEq)]
pub struct Matrix<T> {
    pub rows: usize,
    pub columns: usize,
    elements: Vec<T>
}

impl <T: Clone + Default> Matrix<T> {
    pub fn new(rows: usize, columns: usize) -> Matrix<T> {
        let mut elems = Vec::with_capacity(rows * columns);
        elems.resize(rows * columns, T::default());
        Matrix {
            rows: rows,
            columns: columns,
            elements: elems,
        }
    }

    pub fn t(&self) -> Matrix<T> {
        let mut elems: Vec<T> = Vec::with_capacity(self.rows * self.columns);
        elems.resize(self.rows * self.columns, T::default());
        for row in 0..self.rows {
            for col in 0..self.columns {
                elems[col * self.rows + row] = self.at(row, col);
            }
        }
        Matrix::new_from(self.columns, self.rows, elems)
    }
}

impl<T> Matrix<T> {
    pub fn new_from(rows: usize, columns: usize, elements: Vec<T>) -> Matrix<T> {
        debug_assert!(rows * columns == elements.len());
        Matrix {
            rows: rows,
            columns: columns,
            elements: elements,
        }
    }

    pub fn random<B>(rows: usize, columns: usize, min: B, max: B) -> Matrix<B>
            where B: PartialOrd + rand::distributions::range::SampleRange {
        let between = Range::new(min, max);
        let mut rng = rand::thread_rng();
        let mut elems = Vec::with_capacity(rows * columns);
        for _ in 0..(rows * columns) {
            elems.push(between.ind_sample(&mut rng));
        }
        Matrix {
            rows: rows,
            columns: columns,
            elements: elems,
        }
    }

    pub fn set_at(&mut self, row: usize, column: usize, value: T) {
        debug_assert!(row < self.rows, "row is too large");
        debug_assert!(column < self.columns, "column is too large");
        self.elements[row * self.columns + column] = value
    }

    fn size(&self) -> usize {
        self.rows * self.columns
    }

    pub fn assert_same_size(&self, other: &Matrix<T>) {
        debug_assert!(self.rows == other.rows && self.columns == other.columns,
                "matrix should have same size, given {}x{} and {}x{}",
                self.rows, self.columns, other.rows, other.columns)
    }
}

impl Matrix<usize> {
    pub fn to_one_hot<T: From<u8> + Clone + Default>(&self, classes: usize) -> Matrix<T> {
        debug_assert!(self.columns == 1, "matrix must be Nx1 to change to one_hot");
        let mut matrix = Matrix::new(self.elements.len(), classes);
        for i in 0..self.elements.len() {
            matrix.set_at(i, self.elements[i], T::from(1));
        }
        matrix
    }
}

impl<T: Default + Clone + ops::Add<Output = T> + ops::Mul<Output = T>> Matrix<T> {
    pub fn matmul(&self, other: &Matrix<T>) -> Matrix<T> {
        debug_assert!(self.columns == other.rows, "trying to multiply {}x{} with {}x{}",
            self.rows, self.columns, other.rows, other.columns);
        let mut output = Matrix::new(self.rows, other.columns);
        for row in 0..self.rows {
            for col in 0..other.columns {
                let mut v_ij = T::default();
                for k in 0..self.columns {
                    v_ij = v_ij + self.at(row, k) * other.at(k, col);
                }
                output.set_at(row, col, v_ij);
            }
        }
        output
    }
}

impl<T: Clone> Matrix<T> {
    fn make_mut_op<F>(&mut self, other: &Matrix<T>, mut op: F)
            where F: FnMut(T, T) -> T {
        self.assert_same_size(other);
        for i in 0..self.size() {
            self.elements[i] = op(self.elements[i].clone(), other.elements[i].clone());
        }
    }

    fn make_op<F>(&self, other: &Matrix<T>, op: F) -> Matrix<T>
            where F: FnMut(T, T) -> T {
        let mut output = self.clone();
        output.make_mut_op(other, op);
        output
    }

    pub fn at(&self, row: usize, column: usize) -> T {
        debug_assert!(row < self.rows, "row is too large");
        debug_assert!(column < self.columns, "column is too large");
        self.elements[row * self.columns + column].clone()
    }

    pub fn reduce<F, B>(&self, init: B, mut f: F) -> B
            where F: FnMut(B, T) -> B {
        self.reduce_with_index(init, |acc, v, _row, _col| f(acc, v))
    }

    pub fn reduce_with_index<F, B>(&self, init: B, mut f: F) -> B
            where F: FnMut(B, T, usize, usize) -> B {
        let mut result = init;
        for row in 0..self.rows {
            for col in 0..self.columns {
                result = f(result, self.at(row, col), row, col);
            }
        }
        result
    }

    pub fn reduce_rows_with_index<F, B: Copy>(&self, init: B, mut f: F) -> Matrix<B>
            where F: FnMut(B, T, usize, usize) -> B {
        let mut elems = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut result = init;
            for col in 0..self.columns {
                result = f(result, self.at(row, col), row, col);
            }
            elems.push(result);
        }
        Matrix::new_from(self.rows, 1, elems)
    }

    pub fn reduce_rows<F, B: Copy>(&self, init: B, mut f: F) -> Matrix<B>
            where F: FnMut(B, T) -> B {
        self.reduce_rows_with_index(init, |acc, v, _row, _col| f(acc, v))
    }

    pub fn reduce_columns<F, B: Copy>(&self, init: B, mut f: F) -> Matrix<B>
            where F: FnMut(B, T) -> B {
        self.reduce_columns_with_index(init, |acc, v, _row, _col| f(acc, v))
    }

    pub fn reduce_columns_with_index<F, B: Copy>(&self, init: B, mut f: F) -> Matrix<B>
            where F: FnMut(B, T, usize, usize) -> B {
        let mut elems = Vec::with_capacity(self.columns);
        for col in 0..self.columns {
            let mut result = init;
            for row in 0..self.rows {
                result = f(result, self.at(row, col), row, col);
            }
            elems.push(result);
        }
        Matrix::new_from(1, self.columns, elems)
    }

    pub fn transform<F, B: Default + Clone>(&self, mut f: F) -> Matrix<B>
            where F: FnMut(T) -> B {
        self.transform_with_index(|v, _row, _col| f(v))
    }

    pub fn transform_with_index<F, B: Default + Clone>(&self, mut f: F) -> Matrix<B>
            where F: FnMut(T, usize, usize) -> B {
        let mut output = Matrix::new(self.rows, self.columns);
        for row in 0..self.rows {
            for col in 0..self.columns {
                output.set_at(row, col, f(self.at(row, col), row, col));
            }
        }
        output
    }

    pub fn slice_rows(&self, range: ops::Range<usize>) -> Matrix<T> {
        let rows = range.end - range.start;
        let vec_start = range.start * self.columns;
        let mut elements = Vec::with_capacity(rows * self.columns);
        for i in 0..(rows * self.columns) {
            elements.push(self.elements[i + vec_start].clone());
        }
        Matrix::new_from(rows, self.columns, elements)
    }

    pub fn shuffle_rows(&mut self) -> Vec<(usize, usize)> {
        let between = Range::new(0, self.rows);
        let mut rng = rand::thread_rng();
        let count = self.rows / 2;
        let mut swaps = Vec::with_capacity(count);
        for _ in 0..count {
            let row = between.ind_sample(&mut rng);
            let other = between.ind_sample(&mut rng);
            self.swap_rows(row, other);
            swaps.push((row, other));
        }
        swaps
    }

    pub fn swap_rows(&mut self, row: usize, other_row: usize) {
        for i in 0..self.columns {
            let value = self.at(row, i);
            let other = self.at(other_row, i);
            self.set_at(row, i, other);
            self.set_at(other_row, i, value);
        }
    }
}

impl<T: ops::Add<Output = T> + Clone> Matrix<T> {
    pub fn add_mut(&mut self, other: &Matrix<T>) {
        self.make_mut_op(other, |a, b| a + b)
    }
}

impl<T: ops::Sub<Output = T> + Clone> Matrix<T> {
    pub fn sub_mut(&mut self, other: &Matrix<T>) {
        self.make_mut_op(other, |a, b| a - b)
    }
}

impl<T: ops::Mul<Output = T> + Clone> Matrix<T> {
    pub fn mul_mut(&mut self, other: &Matrix<T>) {
        self.make_mut_op(other, |a, b| a * b)
    }
}

impl<T: ops::Div<Output = T> + Clone> Matrix<T> {
    pub fn div_mut(&mut self, other: &Matrix<T>) {
        self.make_mut_op(other, |a, b| a / b)
    }
}

impl<'a, 'b, T: ops::Add<Output = T> + Clone> ops::Add<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.make_op(&other, |a, b| a + b)
    }
}

impl<'a, 'b, T: ops::Sub<Output = T> + Clone> ops::Sub<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.make_op(&other, |a, b| a - b)
    }
}

impl<'a, 'b, T: ops::Mul<Output = T> + Clone> ops::Mul<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.make_op(&other, |a, b| a * b)
    }
}

impl<'a, 'b, T: ops::Div<Output = T> + Clone> ops::Div<&'b Matrix<T>> for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn div(self, other: &'b Matrix<T>) -> Matrix<T> {
        self.make_op(&other, |a, b| a / b)
    }
}

impl<T: fmt::Display + Clone> fmt::Display for Matrix<T> {
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

impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Matrix<T> {
        Matrix::new_from(self.rows, self.columns, self.elements.to_owned())
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

impl<T> str::FromStr for Matrix<T>
        where T: str::FromStr, <T as str::FromStr>::Err: fmt::Display {
    type Err = ParseMatrixError;
    fn from_str(s: &str) -> Result<Matrix<T>, ParseMatrixError> {
        let lines: Vec<Vec<&str>> = s.trim().split('\n').map(|line| line.split(' ').collect()).collect();
        let rows = lines.len();
        let columns = lines[0].len();
        let mut elems = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for col in 0..columns {
                match T::from_str(lines[row][col]) {
                    Ok(v) => elems.push(v),
                    Err(e) => return Err(ParseMatrixError { message: e.to_string() })
                }
            }
        }
        Ok(Matrix::new_from(rows, columns, elems))
    }
}
