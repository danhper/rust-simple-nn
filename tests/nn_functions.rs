extern crate simple_nn;

use simple_nn::{Matrix, nn};

#[test]
fn functions_softmax() {
    let input = Matrix::new_from(2, 7, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0,
                                            8.0, 3.0, 1.0, 6.0, 12.0, 1.0, 2.0]);
    let output = nn::functions::softmax(&input);
    let expected = vec![
        [2.36405430e-02, 6.42616585e-02, 1.74681299e-01, 4.74833000e-01,
         2.36405430e-02, 6.42616585e-02, 1.74681299e-01],
        [1.79389812e-02, 1.20871905e-04, 1.63582334e-05,
         2.42777710e-03, 9.79435187e-01, 1.63582334e-05, 4.44662887e-05]];
    assert_eq!(output.rows, 2);
    assert_eq!(output.columns, 7);
    for i in 0..7 {
        assert!((output.at(0, i) - expected[0][i]).abs() < 1e-5);
        assert!((output.at(1, i) - expected[1][i]).abs() < 1e-5);
    }
}

#[test]
fn functions_log_softmax() {
    let input = Matrix::new_from(2, 10, vec![
        -1.74014, -1.22728, 0.14055, -0.590518, 0.234221,
        4.78415, -0.90424, -1.31733, 6.42867, 0.0813785,
        0.865385, -2.06612, -1.24789, -1.38634, -1.23427,
        2.53362, -1.325, -1.48494, 0.0346661, 1.28081]);
    let expected = vec![vec![-8.35237491, -7.83951491, -6.47168491, -7.20275291, -6.37801391,
                        -1.82808491, -7.51647491, -7.92956491, -0.18356491, -6.53085641],
                        vec![-2.18184434, -5.11334934, -4.29511934, -4.43356934, -4.28149934,
                        -0.51360934, -4.37222934, -4.53216934, -3.01256324, -1.76641934]];
    let output = nn::functions::log_softmax(&input);
    assert_eq!(output.rows, 2);
    assert_eq!(output.columns, 10);

    for i in 0..10 {
        assert!((output.at(0, i) - expected[0][i]).abs() < 1e-5);
        assert!((output.at(1, i) - expected[1][i]).abs() < 1e-5);
    }
}

#[test]
fn functions_softmax_cross_entropy() {
    let input = Matrix::new_from(2, 10, vec![
        -1.74014, -1.22728, 0.14055, -0.590518, 0.234221,
        4.78415, -0.90424, -1.31733, 6.42867, 0.0813785,
        0.865385, -2.06612, -1.24789, -1.38634, -1.23427,
        2.53362, -1.325, -1.48494, 0.0346661, 1.28081]);

    let labels = Matrix::new_from(2, 1, vec![8, 9]).to_one_hot(10);
    let expected = vec![0.183565, 1.76642];
    let output = nn::functions::softmax_cross_entropy(&input, &labels);
    for i in 0..2 {
        assert!((output.at(i, 0) - expected[i]).abs() < 1e-5);
    }
}

#[test]
fn functions_cross_entropy_from_probs() {
    let input = Matrix::new_from(1, 7, vec![
         0.02364054, 0.06426166, 0.1746813, 0.474833, 0.02364054, 0.06426166, 0.1746813 ]);
    let labels = Matrix::new_from(1, 1, vec![1]).to_one_hot(10);
    let expected = 2.74479212;
    let output = nn::functions::cross_entropy_from_probs(&input, &labels);
    assert!((output.at(0, 0) - expected).abs() < 1e-5);
}

#[test]
fn functions_argmax() {
    let input = Matrix::new_from(2, 4, vec![
        1.0, 2.0, 3.0, 1.0, -1.0, 8.0, 2.0, 1.0
    ]);
    let expected = Matrix::new_from(2, 1, vec![2, 1]);
    let output = nn::functions::argmax(&input);
    assert_eq!(output, expected);
}


#[test]
fn functions_accuracy_from_probs() {
    let input = Matrix::new_from(2, 4, vec![
        0.2, 0.3, 0.4, 0.1, 0.1, 0.7, 0.1, 0.1
    ]);
    let labels: Matrix<usize> = Matrix::new_from(2, 1, vec![1, 1]).to_one_hot(4);
    let accuracy = nn::functions::accuracy_from_probs(&input, &labels);
    assert!((accuracy - 0.5).abs() < 1e-5);
}
