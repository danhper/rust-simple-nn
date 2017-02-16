#[derive(Debug, Default)]
pub struct TrainingResults {
    pub total_count: u64,
    pub current_count: u64,
    pub total_loss: f64,
    pub hit_count: u64,
    pub miss_count: u64
}
