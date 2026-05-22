use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    lr_scheduler::constant::ConstantLr,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LossMetric},
    },
};

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::{Cnn, CnnConfig},
};

// TrainStep is now a trait with associated types (no generic params) in 0.21.
// It's implemented on Cnn<B> where B is an AutodiffBackend.
impl<B: AutodiffBackend> TrainStep for Cnn<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let logits = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());
        let output = ClassificationOutput::new(loss, logits, batch.targets);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

// InferenceStep (renamed from ValidStep) is implemented on the inner (non-autodiff) module.
// The Learner constraint is: M::InnerModule: InferenceStep, so this impl covers validation.
impl<B: Backend> InferenceStep for Cnn<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.targets.clone());
        ClassificationOutput::new(loss, logits, batch.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: CnnConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    let config = TrainingConfig::new(CnnConfig::new(10), AdamConfig::new());

    B::seed(&device, config.seed);

    let dataloader_train = DataLoaderBuilder::<B, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(4)
        .build(MnistDataset::train());

    let dataloader_valid = DataLoaderBuilder::<B::InnerBackend, _, _>::new(MnistBatcher)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(MnistDataset::test());

    let model = config.model.init::<B>(&device);
    let optim = config.optimizer.init();
    let lr = ConstantLr::new(config.learning_rate);

    let learner = Learner::new(model, optim, lr);

    SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary()
        .launch(learner);
}
