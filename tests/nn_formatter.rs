extern crate simple_nn;

use simple_nn::nn::measures;
use simple_nn::nn::training_results::TrainingResults;
use simple_nn::nn::formatter::{ProgressFormatter};

#[test]
fn progress_formatter_add_measure() {
    let mut formatter = ProgressFormatter::new();
    formatter.add_measure(measures::Accuracy::new());
    assert_eq!(formatter.measures.len(), 1)
}

#[test]
fn progress_formatter_format() {
    let mut formatter = ProgressFormatter::new();
    formatter.add_measure(measures::Accuracy::new());
    formatter.add_measure(measures::MeanLoss::new());
    let training_results = TrainingResults {
        total_count: 10,
        current_count: 5,
        total_loss: 123.456789,
        hit_count: 3,
        miss_count: 2
    };
    let formatted = formatter.format(&training_results);
    let expected = "acc = 0.60000, loss = 24.69136";
    assert_eq!(formatted, expected);
}

#[test]
fn progress_formatter_progress() {
    let mut formatter = ProgressFormatter::new();
    formatter.add_measure(measures::Accuracy::new());
    formatter.add_measure(measures::MeanLoss::new());
    let mut training_results = TrainingResults {
        total_count: 20,
        current_count: 10,
        total_loss: 123.456789,
        hit_count: 8,
        miss_count: 2
    };
    let expected = "10 / 20 (50%) [========>---------] - acc = 0.80000, loss = 12.34568";
    assert_eq!(formatter.progress(&training_results), expected);
    training_results.current_count = 20;
    training_results.hit_count = 18;
    let expected_end = "20 / 20 (100%) [=================>] - acc = 0.90000, loss = 6.17284";
    assert_eq!(formatter.progress(&training_results), expected_end);
}
