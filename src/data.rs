use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

// MnistBatch is what we hand to the model during training.
// It holds a whole batch of images and their correct labels as GPU tensors.
#[derive(Clone, Debug)]
pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, 3>,       // shape: [batch_size, 28, 28]
    pub targets: Tensor<B, 1, Int>, // shape: [batch_size]
}

// MnistBatcher converts a Vec of raw MnistItems into a single MnistBatch.
// It stores the device so it knows where to place the tensors (CPU or GPU).
#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<MnistItem, MnistBatch<B>> for MnistBatcher<B> {
    fn batch(&self, items: Vec<MnistItem>) -> MnistBatch<B> {
        // Convert each raw image (28*28 bytes) into a float tensor of shape [1, 28, 28],
        // then normalize pixel values from [0, 255] to [0.0, 1.0].
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<f32>())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 28, 28]) / 255.0)
            .collect();

        // Stack the list of [1, 28, 28] tensors into one [batch_size, 28, 28] tensor.
        let images = Tensor::stack(images, 0);

        // Convert each label (a u8 digit 0-9) into an integer tensor.
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.label as i64], &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
