mod data;
mod model;
mod train;

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};

fn main() {
    let device = WgpuDevice::default();
    train::train::<Autodiff<Wgpu>>("./artifacts", device);
}
