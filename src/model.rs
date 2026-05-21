use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        Linear, LinearConfig, Relu,
    },
    prelude::*,
};

// The model struct holds all the learnable layers.
// #[derive(Module)] is a Burn macro that makes this struct a trainable module —
// it wires up parameter tracking, device movement, and serialization for free.
#[derive(Module, Debug)]
pub struct Cnn<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

// CnnConfig holds the hyperparameters needed to construct the model.
// Separating config from the model makes it easy to save settings to disk
// and to swap values without touching the model code.
#[derive(Config, Debug)]
pub struct CnnConfig {
    num_classes: usize,
}

impl CnnConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Cnn<B> {
        Cnn {
            // Conv layer 1: 1 input channel (grayscale) → 8 feature maps, 3×3 filter
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            // Conv layer 2: 8 input channels → 16 feature maps, 3×3 filter
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            // MaxPool with 2×2 window and stride 2 — halves spatial dimensions
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            // After two conv+pool passes: 16 channels × 5 × 5 spatial = 400
            fc1: LinearConfig::new(16 * 5 * 5, 128).init(device),
            fc2: LinearConfig::new(128, self.num_classes).init(device),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> Cnn<B> {
    // forward() defines what happens when data passes through the network.
    // Input shape:  [batch_size, 1, 28, 28]
    // Output shape: [batch_size, 10]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // [batch, 28, 28] → [batch, 1, 28, 28]  (insert channel dimension)
        let x = x.unsqueeze_dim(1);

        // [batch, 1, 28, 28] → [batch, 8, 26, 26]  (3×3 filter shrinks 28 to 26)
        let x = self.conv1.forward(x);
        // [batch, 8, 26, 26] → [batch, 8, 26, 26]  (relu: element-wise, shape unchanged)
        let x = self.relu.forward(x);
        // [batch, 8, 26, 26] → [batch, 8, 13, 13]  (2×2 pool halves spatial dims)
        let x = self.pool.forward(x);

        // [batch, 8, 13, 13] → [batch, 16, 11, 11]  (3×3 filter shrinks 13 to 11)
        let x = self.conv2.forward(x);
        // [batch, 16, 11, 11] → [batch, 16, 11, 11]  (relu: shape unchanged)
        let x = self.relu.forward(x);
        // [batch, 16, 11, 11] → [batch, 16, 5, 5]  (2×2 pool: floor(11/2) = 5)
        let x = self.pool.forward(x);

        // [batch, 16, 5, 5] → [batch, 400]  (16 × 5 × 5 = 400)
        let [batch, channels, height, width] = x.dims();
        let x = x.reshape([batch, channels * height * width]);

        // [batch, 400] → [batch, 128]
        let x = self.fc1.forward(x);
        // [batch, 128] → [batch, 128]  (relu: shape unchanged)
        let x = self.relu.forward(x);
        // [batch, 128] → [batch, 10]  (one score per digit class)
        let x = self.fc2.forward(x);

        x
    }
}
