use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};

use crate::model::CnnConfig;

pub fn infer<B: Backend>(artifact_dir: &str, image_path: &str, device: B::Device) {
    let record = CompactRecorder::new()
        .load(
            format!("{artifact_dir}/checkpoint/model-10").into(),
            &device,
        )
        .expect("Failed to load model checkpoint — run training first.");

    let model = CnnConfig::new(10).init::<B>(&device).load_record(record);

    // Load image, convert to grayscale, resize to 28×28.
    let img = image::open(image_path)
        .unwrap_or_else(|_| panic!("Could not open image: {image_path}"))
        .to_luma8();

    let img = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);

    // Convert pixels to a [28, 28] float tensor normalised to [0, 1].
    let pixels: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
    let data = TensorData::new(pixels, [28usize, 28]);
    let image_tensor = Tensor::<B, 2>::from_data(data, &device).unsqueeze::<3>();

    // Forward pass → logits [1, 10], then softmax for probabilities.
    let logits = model.forward(image_tensor);
    let probs = burn::tensor::activation::softmax(logits, 1);

    let pred = probs
        .clone()
        .argmax(1)
        .into_scalar()
        .elem::<i64>();

    let confidence = probs
        .slice([0..1, pred as usize..pred as usize + 1])
        .into_scalar()
        .elem::<f32>();

    println!("Predicted digit : {pred}");
    println!("Confidence      : {:.1}%", confidence * 100.0);
}
