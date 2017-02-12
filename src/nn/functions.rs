use linalg::Matrix;

pub fn softmax(matrix: &Matrix) -> Matrix {
    let maxes = matrix.reduce_rows(0.0, |acc, v| if v > acc { v } else { acc });
    let transformed = matrix.transform_with_index(|v, row, _col| (v - maxes.at(row, 0)).exp());
    let sums = transformed.reduce_rows(0.0, |acc, v| acc + v);
    transformed.transform_with_index(|v, row, _col| v / sums.at(row, 0))
}

#[allow(dead_code)]
pub fn log_sum_exp(matrix: &Matrix) -> Matrix {
    let maxes = matrix.reduce_rows(0.0, |acc, v| if v > acc { acc } else { v });
    let transformed = matrix.transform_with_index(|v, row, _col| (v - maxes.at(row, 0)).exp());
    let sums = transformed.reduce_rows(0.0, |acc, v| acc + v);
    maxes.transform_with_index(|v, row, _col| v + sums.at(row, 0).ln())
}

#[allow(dead_code)]
pub fn log_softmax(matrix: &Matrix) -> Matrix {
    let log_z = log_sum_exp(matrix);
    matrix.transform_with_index(|v, row, _col| v - log_z.at(row, 0))
}

#[allow(dead_code)]
pub fn softmax_cross_entropy(matrix: &Matrix, labels: &Matrix) -> Matrix {
    let lsm = log_softmax(matrix);
    lsm.reduce_rows_with_index(0.0, |acc, v, row, col| acc - v * labels.at(row, col))
}
