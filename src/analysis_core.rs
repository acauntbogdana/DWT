//! Основной пайплайн: DWT vs EWMA VaR, DM/GDM тесты.

use std::{
    fs::OpenOptions,
    io::Write,
    time::Instant,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    bars_io::{list_available_tickers, load_bars_for_ticker},
    config::{
        BOOTSTRAP_B, DELTA_REL, EWMA_LAMBDA, HORIZONS, MIN_BARS, MIN_DM_OBS, OUTPUT_DIR,
        VAR_LEVEL, WARMUP_BARS, WINDOW_BARS,
    },
    data_loading::{unix_day_to_date, BarAgg},
    dm_test::{dm_bootstrap_pvalue, dm_statistic, gdm_statistic},
    dwt::{DwtState, OnlineExtrema},
    statistics::median,
    var_models::{dwt_volatility, ewma_variance, quantile_loss, var_forecast},
};

/// Результаты по одному тикеру.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickerResult {
    pub ticker: String,
    /// VaR-тест (DWT vs EWMA): для каждого горизонта (DM, p_нормаль, p_bootstrap).
    pub dm_by_h: Vec<(f64, f64, f64)>,
    /// VaR-тест: обобщённая статистика.
    pub gdm: (f64, f64),
    pub mean_loss_dwt: Vec<f64>,
    pub mean_loss_ewma: Vec<f64>,
    pub n_obs: usize,
    pub date_range: (String, String),

    // === Тесты предсказания среднего (режимной структуры) ===
    /// DWT-режим (μ̂ = среднее по текущему режиму) vs полная история.
    pub dm_mean_vs_full: Vec<(f64, f64, f64)>,
    pub gdm_mean_vs_full: (f64, f64),
    pub mean_mse_dwt: Vec<f64>,
    pub mean_mse_full: Vec<f64>,
     /// DWT-режим vs скользящее окно WINDOW_BARS баров.
    pub dm_mean_vs_window: Vec<(f64, f64, f64)>,
    pub gdm_mean_vs_window: (f64, f64),
    pub mean_mse_window: Vec<f64>,

    // === Диагностика режимной структуры ===
    /// Число смен режима (wiping-out событий) за весь ряд.
    pub n_regimes: usize,
    /// Средняя длина режима в барах (n_obs / (n_regimes + 1)).
    pub mean_regime_len_bars: f64,
}

pub fn run() -> Result<()> {
    std::fs::create_dir_all(OUTPUT_DIR)?;
    let start = Instant::now();

    // Шаг 1: Построение баров (использует кэш state.json).
    println!("Шаг 1: Построение баров для всех тикеров...");
    crate::data_loading::build_all_bars()?;

    // Шаг 2: Список тикеров с барами.
    println!("Шаг 2: Список доступных тикеров...");
    let tickers = list_available_tickers()?;
    println!("Всего тикеров с барами: {}", tickers.len());

    // Шаг 3: Подготовка отчёта.
    let report_path = format!("{}/report.txt", OUTPUT_DIR);
    let mut report = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&report_path)?;
    writeln!(
        report,
        "\n=== DWT vs EWMA VaR анализ от {} ===",
        chrono::Utc::now()
    )?;
    writeln!(
        report,
        "Параметры: δ_rel={:.4}, τ={:.2}, λ={:.2}, горизонты={:?}",
        DELTA_REL, VAR_LEVEL, EWMA_LAMBDA, HORIZONS
    )?;

    // Шаг 4: Анализ по тикерам (с чекпоинтом в tickers_checkpoint.jsonl).
    println!("Шаг 3: Анализ DWT vs EWMA по тикерам...");

    let ckpt_path = format!("{}/tickers_checkpoint.jsonl", OUTPUT_DIR);

    // Читаем уже готовые результаты из прошлых запусков.
    let mut results: Vec<TickerResult> = Vec::new();
    let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Ok(content) = std::fs::read_to_string(&ckpt_path) {
        for line in content.lines() {
            if line.trim().is_empty() { continue; }
            match serde_json::from_str::<TickerResult>(line) {
                Ok(r) => {
                    done.insert(r.ticker.clone());
                    results.push(r);
                }
                Err(e) => eprintln!("Чекпоинт: битая строка пропущена ({})", e),
            }
        }
        println!("Чекпоинт: найдено {} уже обработанных тикеров", done.len());
    }

    let mut ckpt_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&ckpt_path)?;

    let tickers: Vec<String> = tickers
        .into_iter()
        .filter(|t| !done.contains(t))
        .collect();
    println!("Осталось обработать: {} из {}", tickers.len(), tickers.len() + done.len());

    let mut dm_by_h: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];
    let mut gdm_stats: Vec<f64> = Vec::new();
    let mut gdm_pvals: Vec<f64> = Vec::new();
    let mut loss_dwt_by_h: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];
    let mut loss_ewma_by_h: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];

    let mut dm_mean_full_by_h: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];
    let mut gdm_mean_full_stats: Vec<f64> = Vec::new();
    let mut gdm_mean_full_pvals: Vec<f64> = Vec::new();
    let mut dm_mean_window_by_h: Vec<Vec<f64>> = vec![Vec::new(); HORIZONS.len()];
    let mut gdm_mean_window_stats: Vec<f64> = Vec::new();
    let mut gdm_mean_window_pvals: Vec<f64> = Vec::new();

    // Диагностика режимной структуры.
    let mut regimes_per_ticker: Vec<usize> = Vec::new();
    let mut regime_len_bars: Vec<f64> = Vec::new();
    let mut regimes_per_100k: Vec<f64> = Vec::new();

    // Сначала агрегируем уже прочитанные из чекпоинта.
    for res in &results {
        for (h_idx, dm) in res.dm_by_h.iter().enumerate() {
            if dm.0.is_finite() { dm_by_h[h_idx].push(dm.0); }
        }
        if res.gdm.0.is_finite() {
            gdm_stats.push(res.gdm.0);
            gdm_pvals.push(res.gdm.1);
        }
        for (h_idx, v) in res.mean_loss_dwt.iter().enumerate() {
            if v.is_finite() { loss_dwt_by_h[h_idx].push(*v); }
        }
        for (h_idx, v) in res.mean_loss_ewma.iter().enumerate() {
            if v.is_finite() { loss_ewma_by_h[h_idx].push(*v); }
        }
        for (h_idx, dm) in res.dm_mean_vs_full.iter().enumerate() {
            if dm.0.is_finite() { dm_mean_full_by_h[h_idx].push(dm.0); }
        }
        if res.gdm_mean_vs_full.0.is_finite() {
            gdm_mean_full_stats.push(res.gdm_mean_vs_full.0);
            gdm_mean_full_pvals.push(res.gdm_mean_vs_full.1);
        }
        for (h_idx, dm) in res.dm_mean_vs_window.iter().enumerate() {
            if dm.0.is_finite() { dm_mean_window_by_h[h_idx].push(dm.0); }
        }
        if res.gdm_mean_vs_window.0.is_finite() {
            gdm_mean_window_stats.push(res.gdm_mean_vs_window.0);
            gdm_mean_window_pvals.push(res.gdm_mean_vs_window.1);
        }
        if res.n_regimes > 0 && res.n_obs > 0 {
            regimes_per_ticker.push(res.n_regimes);
            regime_len_bars.push(res.mean_regime_len_bars);
            regimes_per_100k.push(res.n_regimes as f64 / res.n_obs as f64 * 100_000.0);
        }
    }

    for (idx, ticker) in tickers.iter().enumerate() {
        let res_opt = match analyze_ticker(ticker) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Тикер {} пропущен: {}", ticker, e);
                None
            }
        };
        if let Some(res) = res_opt {
            for (h_idx, dm) in res.dm_by_h.iter().enumerate() {
                if dm.0.is_finite() { dm_by_h[h_idx].push(dm.0); }
            }
            if res.gdm.0.is_finite() {
                gdm_stats.push(res.gdm.0);
                gdm_pvals.push(res.gdm.1);
            }
            for (h_idx, v) in res.mean_loss_dwt.iter().enumerate() {
                if v.is_finite() { loss_dwt_by_h[h_idx].push(*v); }
            }
            for (h_idx, v) in res.mean_loss_ewma.iter().enumerate() {
                if v.is_finite() { loss_ewma_by_h[h_idx].push(*v); }
            }
            for (h_idx, dm) in res.dm_mean_vs_full.iter().enumerate() {
                if dm.0.is_finite() { dm_mean_full_by_h[h_idx].push(dm.0); }
            }
            if res.gdm_mean_vs_full.0.is_finite() {
                gdm_mean_full_stats.push(res.gdm_mean_vs_full.0);
                gdm_mean_full_pvals.push(res.gdm_mean_vs_full.1);
            }
            for (h_idx, dm) in res.dm_mean_vs_window.iter().enumerate() {
                if dm.0.is_finite() { dm_mean_window_by_h[h_idx].push(dm.0); }
            }
            if res.gdm_mean_vs_window.0.is_finite() {
                gdm_mean_window_stats.push(res.gdm_mean_vs_window.0);
                gdm_mean_window_pvals.push(res.gdm_mean_vs_window.1);
            }
            if res.n_regimes > 0 && res.n_obs > 0 {
                regimes_per_ticker.push(res.n_regimes);
                regime_len_bars.push(res.mean_regime_len_bars);
                regimes_per_100k.push(res.n_regimes as f64 / res.n_obs as f64 * 100_000.0);
            }

            // Чекпоинт: пишем сразу, до push в results.
            if let Ok(json_line) = serde_json::to_string(&res) {
                let _ = writeln!(ckpt_file, "{}", json_line);
                let _ = ckpt_file.flush();
            }

            results.push(res);
        }

        if (idx + 1) % 50 == 0 {
            println!("  обработано {}/{}", idx + 1, tickers.len());
        }
    }

    writeln!(report, "\nПроанализировано тикеров: {}", results.len())?;
    println!("Проанализировано тикеров: {}", results.len());

    // Шаг 5: Сводка DM по горизонтам.
    writeln!(report, "\n--- DM-тест (DWT vs EWMA): d = L_DWT − L_EWMA ---")?;
    writeln!(report, "DM < 0 → DWT точнее; DM > 0 → EWMA точнее.")?;
    for (h_idx, &h) in HORIZONS.iter().enumerate() {
        let dms = &dm_by_h[h_idx];
        if dms.is_empty() {
            writeln!(report, "h={}: нет данных", h)?;
            continue;
        }
        let med = median(dms);
        let frac_dwt_better = dms.iter().filter(|&&x| x < 0.0).count() as f64 / dms.len() as f64;
        let frac_sig = dms.iter().filter(|&&x| x.abs() > 1.96).count() as f64 / dms.len() as f64;
        writeln!(
            report,
            "h={:>3}: медиана DM = {:+.3}, доля DM<0 (DWT лучше) = {:.1}%, \
             доля |DM|>1.96 = {:.1}%, n = {}",
            h,
            med,
            frac_dwt_better * 100.0,
            frac_sig * 100.0,
            dms.len()
        )?;
        // Сравнение средних потерь
        if !loss_dwt_by_h[h_idx].is_empty() && !loss_ewma_by_h[h_idx].is_empty() {
            let m_dwt = median(&loss_dwt_by_h[h_idx]);
            let m_ewma = median(&loss_ewma_by_h[h_idx]);
            writeln!(
                report,
                "    медиана средних потерь: DWT = {:.6}, EWMA = {:.6}",
                m_dwt, m_ewma
            )?;
        }
    }

    // Шаг 6: Сводка GDM.
    writeln!(report, "\n--- GDM (обобщённый DM по всем горизонтам) ---")?;
    if !gdm_stats.is_empty() {
        let med_gdm = median(&gdm_stats);
        let frac_sig = gdm_pvals.iter().filter(|&&p| p < 0.05).count() as f64
            / gdm_pvals.len() as f64;
        writeln!(
            report,
            "Медиана GDM = {:.3}, доля p < 0.05 = {:.1}%, n = {}",
            med_gdm,
            frac_sig * 100.0,
            gdm_stats.len()
        )?;
    } else {
        writeln!(report, "GDM: нет данных")?;
    }

    // === Сводка тестов на предсказание среднего ===
    writeln!(report, "\n=== Тесты на предсказание среднего (режимная структура) ===")?;
    writeln!(report, "d = MSE(DWT-режим) − MSE(альтернатива); d < 0 → DWT-режим точнее.")?;

    writeln!(report, "\n--- DWT-режим vs полная история ---")?;
    for (h_idx, &h) in HORIZONS.iter().enumerate() {
        let dms = &dm_mean_full_by_h[h_idx];
        if dms.is_empty() {
            writeln!(report, "h={}: нет данных", h)?;
            continue;
        }
        let med = median(dms);
        let frac_dwt = dms.iter().filter(|&&x| x < 0.0).count() as f64 / dms.len() as f64;
        let frac_sig = dms.iter().filter(|&&x| x.abs() > 1.96).count() as f64 / dms.len() as f64;
        writeln!(
            report,
            "h={:>3}: медиана DM = {:+.3}, доля DM<0 (DWT-режим лучше) = {:.1}%, \
             доля |DM|>1.96 = {:.1}%, n = {}",
            h, med, frac_dwt * 100.0, frac_sig * 100.0, dms.len()
        )?;
    }
    if !gdm_mean_full_stats.is_empty() {
        let med_gdm = median(&gdm_mean_full_stats);
        let frac_sig = gdm_mean_full_pvals.iter().filter(|&&p| p < 0.05).count() as f64
            / gdm_mean_full_pvals.len() as f64;
        writeln!(
            report,
            "GDM(DWT-режим vs полная история): медиана = {:.3}, доля p < 0.05 = {:.1}%, n = {}",
            med_gdm, frac_sig * 100.0, gdm_mean_full_stats.len()
        )?;
    }

    writeln!(report, "\n--- DWT-режим vs скользящее окно ({} баров) ---", WINDOW_BARS)?;
    for (h_idx, &h) in HORIZONS.iter().enumerate() {
        let dms = &dm_mean_window_by_h[h_idx];
        if dms.is_empty() {
            writeln!(report, "h={}: нет данных", h)?;
            continue;
        }
        let med = median(dms);
        let frac_dwt = dms.iter().filter(|&&x| x < 0.0).count() as f64 / dms.len() as f64;
        let frac_sig = dms.iter().filter(|&&x| x.abs() > 1.96).count() as f64 / dms.len() as f64;
        writeln!(
            report,
            "h={:>3}: медиана DM = {:+.3}, доля DM<0 (DWT-режим лучше) = {:.1}%, \
             доля |DM|>1.96 = {:.1}%, n = {}",
            h, med, frac_dwt * 100.0, frac_sig * 100.0, dms.len()
        )?;
    }
    if !gdm_mean_window_stats.is_empty() {
        let med_gdm = median(&gdm_mean_window_stats);
        let frac_sig = gdm_mean_window_pvals.iter().filter(|&&p| p < 0.05).count() as f64
            / gdm_mean_window_pvals.len() as f64;
        writeln!(
            report,
            "GDM(DWT-режим vs скользящее окно): медиана = {:.3}, доля p < 0.05 = {:.1}%, n = {}",
            med_gdm, frac_sig * 100.0, gdm_mean_window_stats.len()
        )?;
    }

    // === Диагностика режимной структуры ===
    writeln!(report, "\n=== Диагностика режимной структуры (wiping-out) ===")?;
    if !regimes_per_ticker.is_empty() {
        regimes_per_ticker.sort_unstable();
        let n_t = regimes_per_ticker.len();
        let med_reg = regimes_per_ticker[n_t / 2] as f64;
        let med_len = median(&regime_len_bars);
        let med_per_100k = median(&regimes_per_100k);

        let frac_over_500 = regimes_per_100k.iter()
            .filter(|&&x| x >= 500.0).count() as f64 / n_t as f64;
        let frac_under_100 = regimes_per_100k.iter()
            .filter(|&&x| x < 100.0).count() as f64 / n_t as f64;

        writeln!(
            report,
            "Тикеров с диагностикой: {}",
            n_t
        )?;
        writeln!(
            report,
            "Медиана числа режимов на тикер: {:.0}",
            med_reg
        )?;
        writeln!(
            report,
            "Медиана средней длины режима: {:.1} баров",
            med_len
        )?;
        writeln!(
            report,
            "Медиана числа wiping-out на 100 000 баров: {:.1}",
            med_per_100k
        )?;
        writeln!(
            report,
            "Доля тикеров с частотой >= 500 смен на 100k баров: {:.1}%",
            frac_over_500 * 100.0
        )?;
        writeln!(
            report,
            "Доля тикеров с частотой < 100 смен на 100k баров: {:.1}%",
            frac_under_100 * 100.0
        )?;

        // Интерпретационный блок.
        writeln!(report, "\n--- Интерпретация ---")?;
        if med_per_100k >= 500.0 {
            writeln!(
                report,
                "Режимы короткие (>=500 смен на 100k баров). Оценка среднего по режиму \
                 имеет мало наблюдений и высокую дисперсию. Это объясняет отрицательный \
                 результат DWT-режима на среднем: стек слишком часто сбрасывается."
            )?;
        } else if med_per_100k >= 100.0 {
            writeln!(
                report,
                "Режимы средней длины (100-500 смен на 100k баров). Компромисс между \
                 реактивностью и точностью оценки. Результат DWT на среднем может быть \
                 смешанным в зависимости от тикера."
            )?;
        } else {
            writeln!(
                report,
                "Режимы длинные (< 100 смен на 100k баров). Оценка среднего по режиму \
                 должна была бы быть конкурентной. Отрицательный результат DWT на среднем \
                 в этом случае указывает на несовпадение определений: стек размечает \
                 ценовые экстремумы, а не сдвиги среднего."
            )?;
        }
    } else {
        writeln!(report, "Диагностика режимов: нет данных")?;
    }

    // Шаг 7: Графики.
    println!("Шаг 4: Генерация графиков...");
    if let Err(e) = crate::plots::generate_plots(&results) {
        eprintln!("Предупреждение: ошибка построения графиков: {}", e);
    }

    println!("Готово за {:.1} с", start.elapsed().as_secs_f64());
    Ok(())
}

/// Анализ одного тикера.
fn analyze_ticker(ticker: &str) -> Result<Option<TickerResult>> {
    let bars: Vec<BarAgg> = load_bars_for_ticker(ticker)
        .with_context(|| format!("не удалось загрузить бары для {}", ticker))?;
    if bars.len() < MIN_BARS.max(WARMUP_BARS + 100) {
        return Ok(None);
    }

    let n = bars.len();
    let prices: Vec<f64> = bars.iter().map(|b| b.close).collect();
    if prices.iter().any(|p| !p.is_finite() || *p <= 0.0) {
        return Ok(None);
    }

    let log_returns: Vec<f64> = (1..n)
        .map(|i| (prices[i] / prices[i - 1]).ln())
        .collect();

    // Онлайн-DWT: состояние стека и метка режима для каждого бара.
    let mut dwt = OnlineExtrema::new(DELTA_REL);
    let mut states: Vec<DwtState> = Vec::with_capacity(n);
    let mut regime_ids: Vec<usize> = Vec::with_capacity(n);
    let mut regime_id: usize = 0;
    for (i, &p) in prices.iter().enumerate() {
        let (stack, wiped) = dwt.push(p);
        if wiped && i > 0 {
            regime_id += 1;
        }
        let state = DwtState::from_stack(stack, p);
        states.push(state);
        regime_ids.push(regime_id);
    }

    let sigma2_ewma = ewma_variance(&log_returns, EWMA_LAMBDA);

    // === Предсказатели среднего на каждый бар t (без look-ahead) ===
    let mut mu_dwt_regime: Vec<f64> = vec![0.0; n];
    let mut mu_full: Vec<f64> = vec![0.0; n];
    let mut mu_window: Vec<f64> = vec![0.0; n];
    {
        let mut cur_regime_id = regime_ids[0];
        let mut regime_sum = 0.0_f64;
        let mut regime_cnt = 0usize;
        let mut full_sum = 0.0_f64;
        let mut full_cnt = 0usize;
        let mut window_buf: std::collections::VecDeque<f64> =
            std::collections::VecDeque::with_capacity(WINDOW_BARS);
        let mut window_sum = 0.0_f64;

        for t in 0..n {
            mu_dwt_regime[t] = if regime_cnt > 0 { regime_sum / regime_cnt as f64 } else { 0.0 };
            mu_full[t] = if full_cnt > 0 { full_sum / full_cnt as f64 } else { 0.0 };
            mu_window[t] = if !window_buf.is_empty() {
                window_sum / window_buf.len() as f64
            } else {
                0.0
            };

            if t >= 1 && (t - 1) < log_returns.len() {
                let r = log_returns[t - 1];
                regime_sum += r;
                regime_cnt += 1;
                full_sum += r;
                full_cnt += 1;
                if window_buf.len() == WINDOW_BARS {
                    if let Some(old) = window_buf.pop_front() {
                        window_sum -= old;
                    }
                }
                window_buf.push_back(r);
                window_sum += r;

                if regime_ids[t] != cur_regime_id {
                    cur_regime_id = regime_ids[t];
                    regime_sum = 0.0;
                    regime_cnt = 0;
                }
            }
        }
    }

    let d0 = unix_day_to_date(bars[0].date);
    let d1 = unix_day_to_date(bars[n - 1].date);
    let date_range = (
        d0.format("%Y-%m-%d").to_string(),
        d1.format("%Y-%m-%d").to_string(),
    );

    let tau = VAR_LEVEL;
    let mu_h = 0.0;
    let n_h = HORIZONS.len();
    let h_max = HORIZONS.iter().copied().max().unwrap_or(0);

    let mut dm_by_h: Vec<(f64, f64, f64)> = vec![(f64::NAN, f64::NAN, f64::NAN); n_h];
    let mut d_matrix: Vec<Vec<f64>> = vec![Vec::new(); n_h];
    let mut mean_loss_dwt: Vec<f64> = vec![f64::NAN; n_h];
    let mut mean_loss_ewma: Vec<f64> = vec![f64::NAN; n_h];

    let mut dm_mean_vs_full: Vec<(f64, f64, f64)> = vec![(f64::NAN, f64::NAN, f64::NAN); n_h];
    let mut dm_mean_vs_window: Vec<(f64, f64, f64)> = vec![(f64::NAN, f64::NAN, f64::NAN); n_h];
    let mut mean_mse_dwt: Vec<f64> = vec![f64::NAN; n_h];
    let mut mean_mse_full: Vec<f64> = vec![f64::NAN; n_h];
    let mut mean_mse_window: Vec<f64> = vec![f64::NAN; n_h];
    let mut d_matrix_mean_full: Vec<Vec<f64>> = vec![Vec::new(); n_h];
    let mut d_matrix_mean_window: Vec<Vec<f64>> = vec![Vec::new(); n_h];
    let mut sum_mse_dwt: Vec<f64> = vec![0.0; n_h];
    let mut sum_mse_full: Vec<f64> = vec![0.0; n_h];
    let mut sum_mse_window: Vec<f64> = vec![0.0; n_h];

    // valid_count должен жить вне if-блока — он нужен для gdm_statistic ниже.
    let mut valid_count = 0usize;

    // ЕДИНЫЙ диапазон t для всех горизонтов, чтобы d_matrix[h] имели одну длину.
    // log_returns имеет длину n-1. Для горизонта h нужно t+h <= n-1.
    // Требуем t+h_max <= n-1  ⇒  t <= n-1-h_max, т.е. t in [WARMUP, n-h_max).
    if n > WARMUP_BARS + h_max + 1 {
        let t_start = WARMUP_BARS;
        let t_end_excl = n - h_max;

        let mut sum_l_dwt: Vec<f64> = vec![0.0; n_h];
        let mut sum_l_ewma: Vec<f64> = vec![0.0; n_h];

        for t in t_start..t_end_excl {
            let s2 = sigma2_ewma[t];
            if !s2.is_finite() || s2 <= 0.0 {
                continue;
            }
            let sigma_ewma = s2.sqrt();

            let state = &states[t];
            let sigma_dwt = dwt_volatility(state);
            if !sigma_dwt.is_finite() || sigma_dwt <= 0.0 {
                continue;
            }

            let mut d_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut l_dwt_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut l_ewma_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut d_mean_full_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut d_mean_window_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut mse_dwt_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut mse_full_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut mse_window_row: Vec<f64> = Vec::with_capacity(n_h);
            let mut all_valid = true;

            for &h in HORIZONS.iter() {
                debug_assert!(t + h <= log_returns.len());
                let r_forward: f64 = log_returns[t..t + h].iter().sum();
                if !r_forward.is_finite() {
                    all_valid = false;
                    break;
                }

                let var_dwt = var_forecast(mu_h, sigma_dwt, h, tau);
                let var_ewma = var_forecast(mu_h, sigma_ewma, h, tau);
                if !var_dwt.is_finite() || !var_ewma.is_finite() {
                    all_valid = false;
                    break;
                }
                let l_dwt = quantile_loss(r_forward, var_dwt, tau);
                let l_ewma = quantile_loss(r_forward, var_ewma, tau);
                d_row.push(l_dwt - l_ewma);
                l_dwt_row.push(l_dwt);
                l_ewma_row.push(l_ewma);

                let hf = h as f64;
                let pred_dwt = mu_dwt_regime[t] * hf;
                let pred_full = mu_full[t] * hf;
                let pred_window = mu_window[t] * hf;

                let mse_dwt = (r_forward - pred_dwt).powi(2);
                let mse_full = (r_forward - pred_full).powi(2);
                let mse_window = (r_forward - pred_window).powi(2);

                d_mean_full_row.push(mse_dwt - mse_full);
                d_mean_window_row.push(mse_dwt - mse_window);
                mse_dwt_row.push(mse_dwt);
                mse_full_row.push(mse_full);
                mse_window_row.push(mse_window);
            }

            if !all_valid {
                continue;
            }

            for h_idx in 0..n_h {
                d_matrix[h_idx].push(d_row[h_idx]);
                d_matrix_mean_full[h_idx].push(d_mean_full_row[h_idx]);
                d_matrix_mean_window[h_idx].push(d_mean_window_row[h_idx]);
                sum_l_dwt[h_idx] += l_dwt_row[h_idx];
                sum_l_ewma[h_idx] += l_ewma_row[h_idx];
                sum_mse_dwt[h_idx] += mse_dwt_row[h_idx];
                sum_mse_full[h_idx] += mse_full_row[h_idx];
                sum_mse_window[h_idx] += mse_window_row[h_idx];
            }
            valid_count += 1;
        }

        if valid_count >= MIN_DM_OBS {
            for h_idx in 0..n_h {
                if d_matrix[h_idx].len() < MIN_DM_OBS {
                    continue;
                }
                let (dm, p_norm) = dm_statistic(&d_matrix[h_idx]);
                let p_boot = dm_bootstrap_pvalue(&d_matrix[h_idx], BOOTSTRAP_B);
                dm_by_h[h_idx] = (dm, p_norm, p_boot);
                mean_loss_dwt[h_idx] = sum_l_dwt[h_idx] / valid_count as f64;
                mean_loss_ewma[h_idx] = sum_l_ewma[h_idx] / valid_count as f64;

                let (dm_f, p_f) = dm_statistic(&d_matrix_mean_full[h_idx]);
                let pb_f = dm_bootstrap_pvalue(&d_matrix_mean_full[h_idx], BOOTSTRAP_B);
                dm_mean_vs_full[h_idx] = (dm_f, p_f, pb_f);
                mean_mse_dwt[h_idx] = sum_mse_dwt[h_idx] / valid_count as f64;
                mean_mse_full[h_idx] = sum_mse_full[h_idx] / valid_count as f64;

                let (dm_w, p_w) = dm_statistic(&d_matrix_mean_window[h_idx]);
                let pb_w = dm_bootstrap_pvalue(&d_matrix_mean_window[h_idx], BOOTSTRAP_B);
                dm_mean_vs_window[h_idx] = (dm_w, p_w, pb_w);
                mean_mse_window[h_idx] = sum_mse_window[h_idx] / valid_count as f64;
            }
        }
    }

    let gdm = if valid_count >= MIN_DM_OBS {
        gdm_statistic(&d_matrix)
    } else {
        (f64::NAN, f64::NAN)
    };
    let gdm_mean_vs_full = if valid_count >= MIN_DM_OBS {
        gdm_statistic(&d_matrix_mean_full)
    } else {
        (f64::NAN, f64::NAN)
    };
    let gdm_mean_vs_window = if valid_count >= MIN_DM_OBS {
        gdm_statistic(&d_matrix_mean_window)
    } else {
        (f64::NAN, f64::NAN)
    };

    let n_regimes = regime_id;
    let mean_regime_len_bars = if n_regimes > 0 {
        n as f64 / (n_regimes as f64 + 1.0)
    } else {
        n as f64
    };

    Ok(Some(TickerResult {
        ticker: ticker.to_string(),
        dm_by_h,
        gdm,
        mean_loss_dwt,
        mean_loss_ewma,
        n_obs: n,
        date_range,
        dm_mean_vs_full,
        gdm_mean_vs_full,
        mean_mse_dwt,
        mean_mse_full,
        dm_mean_vs_window,
        gdm_mean_vs_window,
        mean_mse_window,
        n_regimes,
        mean_regime_len_bars,
    }))
}
