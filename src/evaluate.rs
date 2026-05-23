use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    prelude::*,
    record::{CompactRecorder, Recorder},
};

use crate::{
    data::{MnistBatch, MnistBatcher},
    model::CnnConfig,
};

pub fn evaluate<B: Backend>(artifact_dir: &str, device: B::Device) {
    let record = CompactRecorder::new()
        .load(
            format!("{artifact_dir}/checkpoint/model-10").into(),
            &device,
        )
        .expect("Failed to load model checkpoint — run training first.");

    let model = CnnConfig::new(10).init::<B>(&device).load_record(record);

    let dataloader = DataLoaderBuilder::<B, _, _>::new(MnistBatcher)
        .batch_size(256)
        .num_workers(4)
        .build(MnistDataset::test());

    // confusion[actual][predicted]
    let mut confusion = [[0u32; 10]; 10];

    for batch in dataloader.iter() {
        let MnistBatch { images, targets } = batch;

        let logits = model.forward(images);
        let predictions = logits.argmax(1).squeeze::<1>();

        let preds: Vec<i32> = predictions.into_data().to_vec().unwrap();
        let labels: Vec<i32> = targets.into_data().to_vec().unwrap();

        for (pred, label) in preds.iter().zip(labels.iter()) {
            confusion[*label as usize][*pred as usize] += 1;
        }
    }

    // Print per-class results.
    println!("\n{:<6} {:>8} {:>8} {:>10}", "Digit", "Correct", "Total", "Accuracy");
    println!("{}", "-".repeat(36));

    let mut total_correct = 0u32;
    let mut total_count = 0u32;

    for digit in 0..10 {
        let correct = confusion[digit][digit];
        let count: u32 = confusion[digit].iter().sum();
        let accuracy = correct as f32 / count as f32 * 100.0;
        println!("{:<6} {:>8} {:>8} {:>9.2}%", digit, correct, count, accuracy);
        total_correct += correct;
        total_count += count;
    }

    println!("{}", "-".repeat(36));
    println!(
        "{:<6} {:>8} {:>8} {:>9.2}%",
        "All",
        total_correct,
        total_count,
        total_correct as f32 / total_count as f32 * 100.0
    );

    // Print confusion matrix.
    println!("\nConfusion matrix (actual → rows, predicted → cols):\n");
    print!("{:>4}  ", "");
    for col in 0..10usize {
        print!("{:>6}", col);
    }
    println!("\n      {}", "------".repeat(10));
    for row in 0..10usize {
        print!("{:>4} |", row);
        for col in 0..10usize {
            print!("{:>6}", confusion[row][col]);
        }
        println!();
    }
}
