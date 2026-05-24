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
            .map(|item| {
                // item.image is [[u8; 28]; 28] — from() preserves the 2D shape,
                // yielding a TensorData of shape [28, 28] with 784 f32 values
                let data = TensorData::from(item.image).convert::<f32>();

                // From the TensorData, create a rank-2 [28, 28] tensor on the target device
                let tensor = Tensor::<B, 2>::from_data(data, device);

                // Normalize pixel values from [0, 255] to [0.0, 1.0]
                tensor / 255.0
            })
            .collect();

        // Tensor::stack(images, 0) inserts a new dimension at position 0 and stack individual [28, 28] tensors along a new batch dimension → [batch_size, 28, 28]. 
        let images = Tensor::stack(images, 0);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([item.label as i64], device))
            .collect();

        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
