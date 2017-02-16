use nn::{layers, objectives, optimizers, measures};
use nn::network::Network;
use nn::formatter::{Formatter, ProgressFormatter};

pub struct NetworkBuilder {
    layers: Vec<Box<layers::Layer>>
}

impl NetworkBuilder<> {
    pub fn new() -> NetworkBuilder {
        NetworkBuilder { layers: vec![] }
    }
}

impl NetworkBuilder {
    pub fn add(mut self, layer: Box<layers::Layer>) -> NetworkBuilder {
        self.layers.push(layer);
        self
    }

    pub fn add_output<Out: layers::OutputLayer>(self, output: Box<Out>) -> NetworkBuilderWithOutput<Out> {
        NetworkBuilderWithOutput { layers: self.layers, output: output }
    }
}

pub struct NetworkBuilderWithOutput<Out: layers::OutputLayer> {
    layers: Vec<Box<layers::Layer>>,
    output: Box<Out>
}

impl <Out: layers::OutputLayer>NetworkBuilderWithOutput<Out> {
    pub fn minimize<Obj: objectives::Objective<Out>>(self, objective: Obj) -> NetworkBuilderWithObjective<Out, Obj>{
        NetworkBuilderWithObjective { layers: self.layers, objective: objective, output: self.output }
    }
}

pub struct NetworkBuilderWithObjective<Out: layers::OutputLayer, Obj: objectives::Objective<Out>> {
    layers: Vec<Box<layers::Layer>>,
    objective: Obj,
    output: Box<Out>
}

impl <Out: layers::OutputLayer, Obj: objectives::Objective<Out>> NetworkBuilderWithObjective<Out, Obj> {
    pub fn with<Opt: optimizers::Optimizer + Clone>(self, optimizer: Opt) -> NetworkBuilderWithOptimizer<Out, Obj, Opt> {
        NetworkBuilderWithOptimizer {
            layers: self.layers, objective: self.objective,
            optimizer: optimizer, output: self.output, formatter: None
        }
    }
}

pub struct NetworkBuilderWithOptimizer<Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone> {
    layers: Vec<Box<layers::Layer>>,
    objective: Obj,
    optimizer: Opt,
    output: Box<Out>,
    formatter: Option<Box<Formatter>>
}

impl <Out: layers::OutputLayer, Obj: objectives::Objective<Out>, Opt: optimizers::Optimizer + Clone> NetworkBuilderWithOptimizer<Out, Obj, Opt> {
    pub fn format_with(mut self, formatter: Box<Formatter>) -> NetworkBuilderWithOptimizer<Out, Obj, Opt> {
        self.formatter = Some(formatter);
        self
    }

    pub fn build(self) -> Network<Out, Obj, Opt> {
        let default_formatter = self.default_formatter();
        let formatter = self.formatter.unwrap_or(Box::new(default_formatter));
        Network::new(self.layers, self.objective, self.optimizer, self.output, formatter)
    }

    pub fn default_formatter(&self) -> ProgressFormatter {
        let mut formatter = ProgressFormatter::new();
        formatter.add_measure(measures::Accuracy::new());
        formatter.add_measure(measures::MeanLoss::new());
        formatter
    }
}
