use nn::measures::Measure;
use nn::training_results::TrainingResults;

pub trait Formatter {
    fn output_results(&self, results: &TrainingResults);
    fn output_epoch_start(&self, epoch: u64, total_epochs: u64);
    fn output_epoch_end(&self, current_epoch: u64, total_epochs: u64);
}

pub struct ProgressFormatter {
    pub measures: Vec<Box<Measure>>,
    pub progress_width: u64
}

impl ProgressFormatter {
    pub fn new() -> ProgressFormatter {
        ProgressFormatter { measures: vec![], progress_width: 40 }
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

impl Formatter for ProgressFormatter {
    fn output_results(&self, results: &TrainingResults) {
        print!("{}\r", self.progress(&results))
    }

    fn output_epoch_start(&self, current_epoch: u64, total_epochs: u64) {
        println!("Training epoch {} / {}", current_epoch, total_epochs);
    }

    fn output_epoch_end(&self, _current_epoch: u64, _total_epochs: u64) {
        println!("")
    }
}
