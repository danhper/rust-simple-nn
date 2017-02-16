use nn::measures::Measure;
use nn::training_results::TrainingResults;

pub struct Formatter {
    pub measures: Vec<Box<Measure>>,
    pub progress_width: u64
}

impl Formatter {
    pub fn new() -> Formatter {
        Formatter { measures: vec![], progress_width: 40 }
    }

    pub fn add_measure(&mut self, measure: Box<Measure>) {
        self.progress_width -= (measure.name().len() + 8) as u64;
        self.measures.push(measure)
    }

    pub fn progress(&self, results: &TrainingResults) -> String {
        let progress = results.current_count * self.progress_width / results.total_count;
        let progress_percentage = results.current_count * 100 / results.total_count;
        let mut bar = String::new();
        for i in 0..(self.progress_width + 1) {
            bar.push_str(match i {
                _ if i == progress => ">",
                _ if i < progress => "=",
                _  => "-",
            });
        }
        format!("{} / {} ({}%) [{}] - {}", results.current_count, results.total_count,
            progress_percentage, bar, self.format(results))
    }

    pub fn format(&self, results: &TrainingResults) -> String {
        self.measures
            .iter()
            .map(|m| format!("{} = {}", m.name(), m.format(m.compute(results))))
            .collect::<Vec<String>>()
            .join(", ")
    }
}
