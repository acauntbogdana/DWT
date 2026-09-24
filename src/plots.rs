//! Графики результатов DWT vs EWMA: распределения DM и GDM.

use anyhow::Result;
use plotters::coord::Shift;
use plotters::prelude::*;
use plotters::style::FontStyle;

use crate::analysis_core::TickerResult;
use crate::config::{HORIZONS, OUTPUT_DIR};

const IMG_WIDTH: u32 = 1400;
const IMG_HEIGHT: u32 = 1100;
const FONT_SIZE: f64 = 46.0;

fn quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let w = pos - lo as f64;
        Some(sorted[lo] + w * (sorted[hi] - sorted[lo]))
    }
}

pub fn generate_plots(results: &[TickerResult]) -> Result<()> {
    if results.is_empty() {
        return Ok(());
    }

    plot_dm_boxplot(results)?;
    plot_gdm_pvalues(results)?;
    plot_gdm_pvalue_cdf(results)?;
    Ok(())
}

/// Boxplot DM-статистик по горизонтам.
fn plot_dm_boxplot(results: &[TickerResult]) -> Result<()> {
    let n_h = HORIZONS.len();
    let mut by_h: Vec<Vec<f64>> = vec![Vec::new(); n_h];
    for r in results {
        for (h_idx, dm) in r.dm_by_h.iter().enumerate() {
            if h_idx < n_h && dm.0.is_finite() {
                by_h[h_idx].push(dm.0);
            }
        }
    }
    let mut data: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::new();
    let mut all_vals: Vec<f64> = Vec::new();
    for (h_idx, _h) in HORIZONS.iter().enumerate() {
        let mut v = by_h[h_idx].clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.len() < 5 {
            continue;
        }
        let q05 = quantile(&v, 0.05).unwrap_or(f64::NAN);
        let q25 = quantile(&v, 0.25).unwrap_or(f64::NAN);
        let q50 = quantile(&v, 0.50).unwrap_or(f64::NAN);
        let q75 = quantile(&v, 0.75).unwrap_or(f64::NAN);
        let q95 = quantile(&v, 0.95).unwrap_or(f64::NAN);
        // Внимание: whiskers — это эмпирические 5% и 95% квантили, а не
        // классические q25 − 1.5·IQR и q75 + 1.5·IQR из boxplot Тьюки.
        // Такая форма устойчивее к тяжёлым хвостам и лучше читается для
        // DM-распределений с экстремальными выбросами по тикерам.
        //
        // x-координата — ИНДЕКС горизонта, не значение.
        data.push((h_idx as f64, q05, q25, q50, q75, q95));
        all_vals.extend_from_slice(&[q05, q95]);
    }
    if data.len() < 2 {
        return Ok(());
    }
    // Гарантируем, что критические линии ±1.96 попадут в кадр,
    // даже если распределение DM сильно смещено.
    let y_min = all_vals
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .min(-2.0);
    let y_max = all_vals
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(2.0);
    let pad = (y_max - y_min).max(0.5) * 0.15;
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;
    let x_lo = -0.5_f64;
    let x_hi = (n_h as f64) - 0.5;

    let png = format!("{}/dm_boxplot.png", OUTPUT_DIR);
    let svg = format!("{}/dm_boxplot.svg", OUTPUT_DIR);

    {
        let root = BitMapBackend::new(&png, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_dm_boxplot(&root, &data, x_lo, x_hi, y_lo, y_hi, n_h)?;
    }
    {
        let root = SVGBackend::new(&svg, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_dm_boxplot(&root, &data, x_lo, x_hi, y_lo, y_hi, n_h)?;
    }
    Ok(())
}

fn draw_dm_boxplot<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    data: &[(f64, f64, f64, f64, f64, f64)],
    x_lo: f64,
    x_hi: f64,
    y_lo: f64,
    y_hi: f64,
    n_h: usize,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            "DM (DWT − EWMA): распределение по горизонтам",
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(150)
        .build_cartesian_2d(x_lo..x_hi, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("Горизонт h (бары)")
        .y_desc("DM")
        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .x_labels(n_h)
        .x_label_formatter(&|x: &f64| {
            let idx = x.round() as isize;
            if idx < 0 || idx as usize >= HORIZONS.len() {
                String::new()
            } else {
                HORIZONS[idx as usize].to_string()
            }
        })
        .draw()?;

    chart.draw_series(LineSeries::new(
        vec![(x_lo, 0.0), (x_hi, 0.0)],
        ShapeStyle::from(&RED).stroke_width(3),
    ))?;

    for &crit in &[-1.96_f64, 1.96_f64] {
        chart.draw_series(LineSeries::new(
            vec![(x_lo, crit), (x_hi, crit)],
            ShapeStyle::from(&BLACK.mix(0.35)).stroke_width(2),
        ))?;
    }

    for (x, q05, q25, q50, q75, q95) in data {
        let x = *x;
        chart.draw_series(LineSeries::new(
            vec![(x, *q05), (x, *q95)],
            ShapeStyle::from(&BLACK).stroke_width(2),
        ))?;
        chart.draw_series(LineSeries::new(
            vec![(x - 0.3, *q05), (x + 0.3, *q05)],
            ShapeStyle::from(&BLACK).stroke_width(2),
        ))?;
        chart.draw_series(LineSeries::new(
            vec![(x - 0.3, *q95), (x + 0.3, *q95)],
            ShapeStyle::from(&BLACK).stroke_width(2),
        ))?;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x - 0.3, *q25), (x + 0.3, *q75)],
            ShapeStyle::from(&BLUE.mix(0.3)).filled(),
        )))?;
        chart.draw_series(LineSeries::new(
            vec![(x - 0.3, *q50), (x + 0.3, *q50)],
            ShapeStyle::from(&BLACK).stroke_width(4),
        ))?;
    }
    root.present()?;
    Ok(())
}

/// Гистограмма p-значений GDM.
fn plot_gdm_pvalues(results: &[TickerResult]) -> Result<()> {
    let pvals: Vec<f64> = results
        .iter()
        .map(|r| r.gdm.1)
        .filter(|p| p.is_finite())
        .collect();
    if pvals.len() < 5 {
        return Ok(());
    }
    let bins = 20usize;
    let mut counts = vec![0usize; bins];
    for &p in &pvals {
        let idx = ((p * bins as f64).floor() as usize).min(bins - 1);
        counts[idx] += 1;
    }
    let max_count = *counts.iter().max().unwrap_or(&1) as f64;
    let y_hi = (max_count * 1.1).max(1.0);

    let png = format!("{}/gdm_pvalues.png", OUTPUT_DIR);
    let svg = format!("{}/gdm_pvalues.svg", OUTPUT_DIR);

    {
        let root = BitMapBackend::new(&png, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_gdm_pvalues(&root, &counts, bins, y_hi)?;
    }
    {
        let root = SVGBackend::new(&svg, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_gdm_pvalues(&root, &counts, bins, y_hi)?;
    }
    Ok(())
}

fn draw_gdm_pvalues<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    counts: &[usize],
    bins: usize,
    y_hi: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            "Распределение p-value GDM по тикерам",
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(150)
        .build_cartesian_2d(0.0_f64..1.0_f64, 0.0_f64..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("p-value")
        .y_desc("Число тикеров")
        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    chart.draw_series(
        counts
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let x0 = i as f64 / bins as f64;
                let x1 = (i + 1) as f64 / bins as f64;
                Rectangle::new(
                    [(x0, 0.0), (x1, c as f64)],
                    ShapeStyle::from(&BLUE.mix(0.6)).filled(),
                )
            }),
    )?;

    chart.draw_series(LineSeries::new(
        vec![(0.05, 0.0), (0.05, y_hi)],
        ShapeStyle::from(&RED).stroke_width(3),
    ))?;

    root.present()?;
    Ok(())
}

/// Эмпирическая CDF p-значений GDM.
///
/// В отличие от гистограммы, сразу показывает долю тикеров с p < α
/// при любом α: значение CDF в точке α = 0.05 — это и есть доля
/// значимых отклонений нулевой гипотезы E[d_{t,h}] = 0 для всех h.
fn plot_gdm_pvalue_cdf(results: &[TickerResult]) -> Result<()> {
    let mut pvals: Vec<f64> = results
        .iter()
        .map(|r| r.gdm.1)
        .filter(|p| p.is_finite())
        .collect();
    if pvals.len() < 5 {
        return Ok(());
    }
    pvals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Эмпирическая CDF: ступенчатая функция, по одной точке на наблюдение.
    // Левая граница — (0, 0), правая — (1, 1), чтобы кривая начиналась
    // в нуле и доходила до единицы.
    let n = pvals.len() as f64;
    let mut cdf: Vec<(f64, f64)> = Vec::with_capacity(pvals.len() + 2);
    // Настоящая эмпирическая CDF — ступенчатая функция:
    // слева от p_i значение i/n, справа (i+1)/n. Vertical-сегменты
    // LineSeries нарисует сам.
    cdf.push((0.0, 0.0));
    for (i, &p) in pvals.iter().enumerate() {
        cdf.push((p, i as f64 / n));
        cdf.push((p, (i as f64 + 1.0) / n));
    }
    cdf.push((1.0, 1.0));

    // Доля p < 0.05 — то же число, что уже выводится в report.txt,
    // но здесь сразу видно графически.
    let frac_sig_05 = pvals.iter().filter(|&&p| p < 0.05).count() as f64 / n;
    let frac_sig_01 = pvals.iter().filter(|&&p| p < 0.01).count() as f64 / n;

    let png = format!("{}/gdm_pvalue_cdf.png", OUTPUT_DIR);
    let svg = format!("{}/gdm_pvalue_cdf.svg", OUTPUT_DIR);

    {
        let root = BitMapBackend::new(&png, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_gdm_pvalue_cdf(&root, &cdf, frac_sig_05, frac_sig_01)?;
    }
    {
        let root = SVGBackend::new(&svg, (IMG_WIDTH, IMG_HEIGHT)).into_drawing_area();
        draw_gdm_pvalue_cdf(&root, &cdf, frac_sig_05, frac_sig_01)?;
    }
    Ok(())
}

fn draw_gdm_pvalue_cdf<DB: DrawingBackend>(
    root: &DrawingArea<DB, Shift>,
    cdf: &[(f64, f64)],
    frac_sig_05: f64,
    frac_sig_01: f64,
) -> Result<()>
where
    DB::ErrorType: 'static,
{
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(root)
        .caption(
            "Эмпирическая CDF p-value GDM",
            FontDesc::new(FontFamily::SansSerif, FONT_SIZE, FontStyle::Normal),
        )
        .margin(40)
        .x_label_area_size(90)
        .y_label_area_size(150)
        .build_cartesian_2d(0.0_f64..1.0_f64, 0.0_f64..1.0_f64)?;

    chart
        .configure_mesh()
        .x_desc("p-value")
        .y_desc("Доля тикеров с p ≤ x")
        .label_style(FontDesc::new(FontFamily::SansSerif, 34.0, FontStyle::Normal))
        .axis_style(ShapeStyle::from(&BLACK).stroke_width(3))
        .draw()?;

    for &alpha in &[0.05_f64, 0.01_f64] {
        chart.draw_series(LineSeries::new(
            vec![(alpha, 0.0), (alpha, 1.0)],
            ShapeStyle::from(&RED.mix(0.55)).stroke_width(2),
        ))?;
    }
    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (1.0, 1.0)],
        ShapeStyle::from(&BLACK.mix(0.25)).stroke_width(2),
    ))?;

    chart.draw_series(LineSeries::new(
        cdf.iter().copied(),
        ShapeStyle::from(&BLUE).stroke_width(4),
    ))?;

    let y05 = (frac_sig_05 + 0.04).min(0.92);
    let y01 = (frac_sig_01 + 0.04).min(0.84);
    let label_05 = format!("{:.1}% p<0.05", frac_sig_05 * 100.0);
    let label_01 = format!("{:.1}% p<0.01", frac_sig_01 * 100.0);
    chart.draw_series(std::iter::once(Text::new(
        label_05,
        (0.06, y05),
        FontDesc::new(FontFamily::SansSerif, 30.0, FontStyle::Normal)
            .color(&RED.mix(0.9)),
    )))?;
    chart.draw_series(std::iter::once(Text::new(
        label_01,
        (0.02, y01),
        FontDesc::new(FontFamily::SansSerif, 30.0, FontStyle::Normal)
            .color(&RED.mix(0.9)),
    )))?;

    root.present()?;
    Ok(())
}