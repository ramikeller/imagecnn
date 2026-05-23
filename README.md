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

## Evaluation

Per-class accuracy on the 10,000-image MNIST test set:

| Digit | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| 0     | 971     | 980   | 99.08%   |
| 1     | 1127    | 1135  | 99.30%   |
| 2     | 993     | 1032  | 96.22%   |
| 3     | 993     | 1010  | 98.32%   |
| 4     | 965     | 982   | 98.27%   |
| 5     | 872     | 892   | 97.76%   |
| 6     | 934     | 958   | 97.49%   |
| 7     | 995     | 1028  | 96.79%   |
| 8     | 955     | 974   | 98.05%   |
| 9     | 967     | 1009  | 95.84%   |
| **All** | **9772** | **10000** | **97.72%** |

Confusion matrix (actual → rows, predicted → cols):

```
        0     1     2     3     4     5     6     7     8     9
   0 |   971     0     0     1     0     1     2     1     4     0
   1 |     0  1127     3     1     0     0     2     0     2     0
   2 |     3     3   993    12     1     1     0    11     8     0
   3 |     1     0     1   993     0     3     0     3     8     1
   4 |     1     0     0     0   965     0     1     2     2    11
   5 |     2     0     0     9     0   872     3     1     5     0
   6 |     7     2     1     0     6     6   934     0     2     0
   7 |     1     4    11     8     0     0     0   995     2     7
   8 |     4     0     1     4     3     1     1     2   955     3
   9 |     6     5     0     6     9     2     0     6     8   967
```

Notable confusions: 9→4 (11), 2→3 (12), 2→7 (11), 7→2 (11).

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

**Evaluate on the full test set:**

```bash
cargo run --release -- evaluate
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
  infer.rs    — loads checkpoint and classifies a single image
  evaluate.rs — runs the full test set and prints per-class accuracy + confusion matrix
  main.rs   — clap CLI entry point (WGPU backend)
```
