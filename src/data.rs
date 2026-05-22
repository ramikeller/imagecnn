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

// In 0.21, Batcher no longer stores a device — the device is passed at batch time.
#[derive(Clone)]
pub struct MnistBatcher;

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<f32>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]) / 255.0)
            .collect();

        let images = Tensor::stack(images, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.label as i64], device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
