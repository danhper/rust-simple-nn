use nn::{layers, objectives};

pub struct FinalLayer {
    pub layer: Box<layers::Layer>,
    pub objective: Box<objectives::Objective>
}

impl FinalLayer {
    pub fn new(layer: Box<layers::Layer>, objective: Box<objectives::Objective>) -> FinalLayer {
        FinalLayer {
            layer: layer,
            objective: objective
        }
    }
}

pub trait OutputLayer<T: objectives::Objective> {
    fn minimizing(&self, objective: T) -> FinalLayer;
}

impl OutputLayer<objectives::CrossEntropy> for layers::Softmax {
    fn minimizing(&self, mut objective: objectives::CrossEntropy) -> FinalLayer {
        objective.set_activation(Box::new(self.clone()));
        FinalLayer::new(Box::new(self.clone()), Box::new(objective))
    }
}
