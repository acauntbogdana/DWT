// Точка входа в приложение.

use std::time::Instant;
use anyhow::Result;
use dataset_report::analysis_core;

fn main() -> Result<()> {
    let start = Instant::now();
    analysis_core::run()?;
    println!("Общее время: {:.1} с", start.elapsed().as_secs_f32());
    Ok(())
}