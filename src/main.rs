mod data;
mod infer;
mod model;
mod train;

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "CNN for MNIST digit classification")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train the model and save checkpoints to ./artifacts
    Train,
    /// Classify a digit image using the trained model
    Infer {
        /// Path to a PNG or JPEG image (will be resized to 28×28)
        image: String,
    },
}

fn main() {
    let cli = Cli::parse();
    let device = WgpuDevice::default();

    match cli.command {
        Command::Train => train::train::<Autodiff<Wgpu>>("./artifacts", device),
        Command::Infer { image } => infer::infer::<Wgpu>("./artifacts", &image, device),
    }
}
