//! Модели VaR: EWMA (RiskMetrics) и DWT-conditional.

use crate::dwt::DwtState;
use statrs::distribution::{ContinuousCDF, Normal};

/// Квантиль стандартного нормального распределения.
pub fn z_alpha(alpha: f64) -> f64 {
    Normal::new(0.0, 1.0)
        .expect("нормальное распределение")
        .inverse_cdf(alpha)
}

/// EWMA-дисперсия (RiskMetrics, формула (12) из статьи).
/// Возвращает вектор σ̂²_t (для каждого t используется информация до t-1).
pub fn ewma_variance(returns: &[f64], lambda: f64) -> Vec<f64> {
    let n = returns.len();
    let mut sigma2 = vec![f64::NAN; n];
    if n == 0 {
        return sigma2;
    }
    // Инициализация: используем только первое наблюдение — никакого
    // заглядывания вперёд. При WARMUP_BARS = 200 и λ = 0.94 рекурсия к
    // моменту t = WARMUP_BARS полностью забывает начальное условие
    // (λ^200 ≈ 5·10⁻⁶), поэтому конкретный выбор σ²₀ не влияет на
    // результаты теста. Формула σ²₀ в исходной статье RiskMetrics не
    // зафиксирована и в разных реализациях варьируется.
    let mut s2 = returns[0].powi(2).max(1e-12);
    for t in 0..n {
        sigma2[t] = s2;
        s2 = (1.0 - lambda) * returns[t].powi(2) + lambda * s2;
        if !s2.is_finite() || s2 <= 0.0 {
            s2 = 1e-12;
        }
    }
    sigma2
}

/// 1-барная волатильность, извлечённая из DWT-стека.
///
/// RMS-амплитуда между соседними выжившими разворотами — это σ движения
/// за `mean_gap_bars` баров. Чтобы получить сопоставимую с EWMA
/// 1-барную волатильность, делим на sqrt(mean_gap_bars).
pub fn dwt_volatility(state: &DwtState) -> f64 {
    let amp = if state.rms_amplitude.is_finite() && state.rms_amplitude > 0.0 {
        state.rms_amplitude
    } else if state.mean_amplitude.is_finite() && state.mean_amplitude > 0.0 {
        state.mean_amplitude
    } else {
        return f64::NAN;
    };
    if state.mean_gap_bars.is_finite() && state.mean_gap_bars > 0.0 {
        amp / state.mean_gap_bars.sqrt()
    } else {
        amp
    }
}

/// VaR-прогноз по заданной σ_t на горизонт `h` (формула (13)).
/// `mu_h` — средний h-периодный доход; `sigma` — σ̂_t (в лог-доходностях).
pub fn var_forecast(mu_h: f64, sigma: f64, h: usize, tau: f64) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 {
        return f64::NAN;
    }
    mu_h + z_alpha(tau) * sigma * (h as f64).sqrt()
}

/// Квантильная функция потерь (формула (16) из статьи) — стандартная
/// pinball-потеря Кенкера–Бассета [12]:
/// L_τ(y, VaR) = max{ τ·(y − VaR), (τ−1)·(y − VaR) }.
///
/// При τ = 0.05:
///   y < VaR (пробой 5%-VaR)  → вес (1−τ) = 0.95 — сильный штраф;
///   y ≥ VaR (нет пробоя)     → вес τ = 0.05 — слабый штраф.
///
/// Замечание: в Wang et al. (2024, ур. 11) эта же функция приведена с
/// обратным знаком разности (VaR − y); при τ = 0.05 такая запись
/// оценивает (1−τ)-квантиль, что противоречит и определению VaR, и
/// собственным результатам бэктеста Wang et al. (HR ≈ τ). Здесь и в
/// статье используется каноническая форма (y − VaR).
pub fn quantile_loss(y: f64, var: f64, tau: f64) -> f64 {
    let diff = y - var;
    let a = tau * diff;
    let b = (tau - 1.0) * diff;
    if a > b {
        a
    } else {
        b
    }
}