//! Diebold-Mariano и Generalized DM тесты.
//!
//! HAC-оценка: QS-ядро (Andrews 1991) + VAR(1)-prewhitening.
//! Bootstrap: moving-block с длиной блока ~ N^(1/3).

use rand::Rng;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::config::QS_KERNEL_COEF;

/// QS-ядро: K(x) = 3/x² · (sin x / x − cos x).
fn qs_kernel(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    3.0 / (x * x) * (x.sin() / x - x.cos())
}

/// HAC-дисперсия скалярного ряда с QS-ядром и заданной полосой.
pub fn hac_variance_qs(x: &[f64], bw: f64) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let c: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let bw = bw.max(1.0);
    let max_lag = ((2.0 * bw).ceil() as usize).min(n - 1).max(1);
    let mut s = 0.0;
    for j in 0..=max_lag {
        let xarg = 6.0 * std::f64::consts::PI * j as f64 / (5.0 * bw);
        let w = qs_kernel(xarg);
        let mut g = 0.0;
        for t in 0..(n - j) {
            g += c[t] * c[t + j];
        }
        g /= n as f64;
        s += if j == 0 { w * g } else { 2.0 * w * g };
    }
    s
}

/// AR(1)-prewhitening: x_t = c + ρ x_{t-1} + e_t.
/// Возвращает (ρ̂, остатки e).
pub fn var1_prewhiten(x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len();
    if n < 3 {
        return (0.0, x.to_vec());
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let c: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 1..n {
        num += c[t] * c[t - 1];
        den += c[t - 1] * c[t - 1];
    }
    let rho = if den.abs() > 1e-12 { num / den } else { 0.0 };
    let rho = rho.clamp(-0.999, 0.999);
    let mut e = Vec::with_capacity(n - 1);
    for t in 1..n {
        e.push(c[t] - rho * c[t - 1]);
    }
    (rho, e)
}

/// Долгосрочная дисперсия скалярного ряда:
/// HAC QS по остаткам + recoloring Ψ = σ²_e / (1 − ρ̂)².
pub fn long_run_variance(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 5 {
        return hac_variance_qs(x, 1.0);
    }
    let (rho, e) = var1_prewhiten(x);
    let bw = QS_KERNEL_COEF * (n as f64).powf(0.2);
    let var_e = hac_variance_qs(&e, bw);
    var_e / (1.0 - rho).powi(2)
}

/// Индивидуальный DM-тест для одного горизонта.
/// `d` — ряд дифференциалов потерь (DWT − EWMA).
/// Возвращает (DM, p-value по нормали).
pub fn dm_statistic(d: &[f64]) -> (f64, f64) {
    let n = d.len();
    if n < 5 {
        return (f64::NAN, f64::NAN);
    }
    let mean_d = d.iter().sum::<f64>() / n as f64;
    let var_lr = long_run_variance(d);
    if !var_lr.is_finite() || var_lr <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let se = (var_lr / n as f64).sqrt();
    let dm = mean_d / se;
    let nd = Normal::new(0.0, 1.0).expect("N(0,1)");
    let p = 2.0 * (1.0 - nd.cdf(dm.abs()));
    (dm, p)
}

/// Кросс-ковариация двух рядов с HAC QS.
fn cross_hac(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return f64::NAN;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let bw = QS_KERNEL_COEF * (n as f64).powf(0.2);
    let bw = bw.max(1.0);
    let max_lag = ((2.0 * bw).ceil() as usize).min(n - 1).max(1);
    let mut s = 0.0;
    for j in 0..=max_lag {
        let xarg = 6.0 * std::f64::consts::PI * j as f64 / (5.0 * bw);
        let w = qs_kernel(xarg);
        let mut g_fwd = 0.0;
        let mut g_bwd = 0.0;
        for t in 0..(n - j) {
            g_fwd += (x[t] - mx) * (y[t + j] - my);
            g_bwd += (y[t] - my) * (x[t + j] - mx);
        }
        g_fwd /= n as f64;
        g_bwd /= n as f64;
        s += if j == 0 { w * g_fwd } else { w * (g_fwd + g_bwd) };
    }
    s
}

/// GDM-статистика (формула (20)–(24)).
/// `d_mat[i]` — ряд дифференциалов для горизонта `i`.
pub fn gdm_statistic(d_mat: &[Vec<f64>]) -> (f64, f64) {
    let h = d_mat.len();
    if h == 0 {
        return (f64::NAN, f64::NAN);
    }
    let t = d_mat[0].len();
    if t < 10 || d_mat.iter().any(|v| v.len() != t) {
        return (f64::NAN, f64::NAN);
    }
    let means: Vec<f64> = d_mat
        .iter()
        .map(|row| row.iter().sum::<f64>() / t as f64)
        .collect();
    let mut psi = vec![vec![0.0_f64; h]; h];
    for i in 0..h {
        for j in 0..h {
            psi[i][j] = if i == j {
                long_run_variance(&d_mat[i])
            } else {
                cross_hac(&d_mat[i], &d_mat[j])
            };
        }
    }
    let psi_mat = nalgebra::DMatrix::from_fn(h, h, |i, j| psi[i][j]);
    let means_vec = nalgebra::DVector::from_vec(means);
    let psi_inv = match psi_mat.try_inverse() {
        Some(inv) => inv,
        None => return (f64::NAN, f64::NAN),
    };
    let quad = means_vec.transpose() * psi_inv * means_vec;
    let gdm = t as f64 * quad[(0, 0)];
    if !gdm.is_finite() || gdm < 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let chi2 = ChiSquared::new(h as f64).expect("chi2");
    let p = 1.0 - chi2.cdf(gdm);
    (gdm, p)
}

/// Moving-block bootstrap p-value для DM.
pub fn dm_bootstrap_pvalue(d: &[f64], b: usize) -> f64 {
    let n = d.len();
    if n < 20 {
        return f64::NAN;
    }
    let block = ((n as f64).powf(1.0 / 3.0)).ceil() as usize;
    let block = block.clamp(2, n);

    let mean_d = d.iter().sum::<f64>() / n as f64;
    let c: Vec<f64> = d.iter().map(|v| v - mean_d).collect();

    let var_lr = long_run_variance(&c);
    if !var_lr.is_finite() || var_lr <= 0.0 {
        return f64::NAN;
    }
    let se_obs = (var_lr / n as f64).sqrt();
    let dm_obs = mean_d / se_obs;

    let mut rng = rand::thread_rng();
    let num_blocks = (n + block - 1) / block;
    let mut count = 0usize;
    for _ in 0..b {
        let mut sample = Vec::with_capacity(n);
        for _ in 0..num_blocks {
            let start = rng.gen_range(0..=(n - block));
            sample.extend_from_slice(&c[start..start + block]);
        }
        sample.truncate(n);
        let mean_s = sample.iter().sum::<f64>() / n as f64;
        let var_s = long_run_variance(&sample);
        if !var_s.is_finite() || var_s <= 0.0 {
            continue;
        }
        let se_s = (var_s / n as f64).sqrt();
        let dm_s = mean_s / se_s;
        if dm_s.abs() >= dm_obs.abs() {
            count += 1;
        }
    }
    // Add-one коррекция (Davison & Hinkley 1997, гл. 4):
    // гарантирует p > 0 и убирает смещение при малых B.
    (count + 1) as f64 / (b + 1) as f64
}