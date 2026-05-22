# imagecnn

A convolutional neural network trained on the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit dataset, written in Rust using the [Burn](https://burn.dev) deep learning framework (v0.21).

## Architecture

```
Input [batch, 28, 28]
  → Conv2d(1→8, 3×3) → ReLU → MaxPool2d(2×2)   # [batch, 8, 13, 13]
  → Conv2d(8→16, 3×3) → ReLU → MaxPool2d(2×2)  # [batch, 16, 5, 5]
  → Flatten                                      # [batch, 400]
  → Linear(400→128) → ReLU                      # [batch, 128]
  → Linear(128→10)                              # [batch, 10]
```

Total parameters: **53,866**

## Results

After 10 epochs (batch size 64, Adam lr=1e-4):

| Split      | Accuracy | Loss  |
|------------|----------|-------|
| Train      | 97.56%   | 0.082 |
| Validation | 97.75%   | 0.071 |

## Usage

**Train:**

```bash
cargo run --release -- train
```

Checkpoints and metrics are written to `./artifacts/`.

**Infer:**

```bash
cargo run --release -- infer path/to/digit.png
```

Accepts any PNG or JPEG. The image is resized to 28×28 grayscale before classification.

```
Predicted digit : 7
Confidence      : 99.3%
```

**Help:**

```bash
cargo run --release -- --help
cargo run --release -- infer --help
```

## Project structure

```
src/
  data.rs   — MNIST batcher (normalises pixels to [0, 1])
  model.rs  — CNN definition
  train.rs  — training loop, optimizer, metrics
  infer.rs  — loads checkpoint and classifies a single image
  main.rs   — clap CLI entry point (WGPU backend)
```
