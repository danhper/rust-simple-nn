use linalg::Matrix;

pub fn softmax(matrix: &Matrix) -> Matrix {
    let mut output = Matrix::new(matrix.rows, matrix.columns);
    for col in 0..matrix.columns {
        let mut sum = 0.0;
        for row in 0..matrix.rows {
            sum += matrix.at(row, col).exp();
        }
        for row in 0..matrix.rows {
            output.set_at(row, col, matrix.at(row, col).exp() / sum);
        }
    }
    output
}
