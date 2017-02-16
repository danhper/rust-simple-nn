use nn::training_results::TrainingResults;

pub trait Measure {
    fn name(&self) -> String;
    fn compute(&self, results: &TrainingResults) -> f64;
    fn format(&self, result: f64) -> String {
        format!("{:.5}", result)
    }
}

pub struct Accuracy;

impl Accuracy {
    pub fn new() -> Box<Accuracy> {
        Box::new(Accuracy {})
    }
}

impl Measure for Accuracy {
    fn name(&self) -> String {
        String::from("acc")
    }

    fn compute(&self, results: &TrainingResults) -> f64 {
        (results.hit_count as f64) / (results.current_count as f64)
    }
}

pub struct MeanLoss;

impl MeanLoss {
    pub fn new() -> Box<MeanLoss> {
        Box::new(MeanLoss {})
    }
}

impl Measure for MeanLoss {
    fn name(&self) -> String {
        String::from("loss")
    }

    fn compute(&self, results: &TrainingResults) -> f64 {
        (results.total_loss) / (results.current_count as f64)
    }
}
