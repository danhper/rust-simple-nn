extern crate simple_nn;

use simple_nn::{Matrix, nn};

#[test]
fn functions_softmax() {
    let input = Matrix::new_from(7, 1, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]);
    let output = nn::functions::softmax(&input);
    let expected = vec![0.02364054, 0.06426166, 0.1746813, 0.474833,  0.02364054, 0.06426166, 0.1746813];
    assert_eq!(output.rows, 7);
    assert_eq!(output.columns, 1);
    for i in 0..7 {
        assert!((output.at(i, 0) - expected[i]).abs() < 1e-5);
    }
}
