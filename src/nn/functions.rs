use linalg::Matrix;

pub fn softmax(matrix: &Matrix<f64>) -> Matrix<f64> {
    let maxes = matrix.reduce_rows(0.0, |acc, v| if v > acc { v } else { acc });
    let transformed = matrix.transform_with_index(|v, row, _col| (v - maxes.at(row, 0)).exp());
    let sums = transformed.reduce_rows(0.0, |acc, v| acc + v);
    transformed.transform_with_index(|v, row, _col| v / sums.at(row, 0))
}

pub fn log_softmax(matrix: &Matrix<f64>) -> Matrix<f64> {
    softmax(matrix).transform(|v| v.ln())
}

pub fn softmax_cross_entropy(matrix: &Matrix<f64>, labels: &Matrix<f64>) -> Matrix<f64> {
    let lsm = log_softmax(matrix);
    lsm.reduce_rows_with_index(0.0, |acc, v, row, col| acc - v * labels.at(row, col))
}

pub fn cross_entropy_from_probs(matrix: &Matrix<f64>, labels: &Matrix<f64>) -> Matrix<f64> {
    matrix.reduce_rows_with_index(0.0, |acc, v, row, col| {
        let label = labels.at(row, col);
        if label > 0.0 { acc - v.ln() } else { acc }
    })
}

pub fn argmax<T: Copy + Default + PartialOrd>(matrix: &Matrix<T>) -> Matrix<u8> {
    matrix.reduce_rows_with_index((T::default(), 0), |(max, max_index), v, _row, col| {
        if v > max { (v, col) } else { (max, max_index) }
    }).transform(|v| v.1 as u8)
}
