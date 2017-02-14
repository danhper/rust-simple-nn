use linalg::Matrix;

pub fn softmax(matrix: &Matrix) -> Matrix {
    let maxes = matrix.reduce_rows(0.0, |acc, v| if v > acc { v } else { acc });
    let transformed = matrix.transform_with_index(|v, row, _col| (v - maxes.at(row, 0)).exp());
    let sums = transformed.reduce_rows(0.0, |acc, v| acc + v);
    transformed.transform_with_index(|v, row, _col| v / sums.at(row, 0))
}

pub fn log_softmax(matrix: &Matrix) -> Matrix {
    softmax(matrix).transform(|v| v.ln())
}

pub fn softmax_cross_entropy(matrix: &Matrix, labels: &Matrix) -> Matrix {
    let lsm = log_softmax(matrix);
    lsm.reduce_rows_with_index(0.0, |acc, v, row, col| acc - v * labels.at(row, col))
}

pub fn cross_entropy_from_probs(matrix: &Matrix, labels: &Matrix) -> Matrix {
    matrix.reduce_rows_with_index(0.0, |acc, v, row, col| {
        let label = labels.at(row, col);
        if label > 0.0 { acc - v.ln() } else { acc }
    })
}
