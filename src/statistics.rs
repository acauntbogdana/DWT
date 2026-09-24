//! Мелкие статистические утилиты, используемые в отчёте.
//!
//! NaN и бесконечности игнорируются во всех функциях.

/// Медиана, игнорируя NaN/±inf. Возвращает `f64::NAN`, если данных нет.
pub fn median(data: &[f64]) -> f64 {
    let mut filtered: Vec<f64> = data.iter().copied().filter(|x| x.is_finite()).collect();
    if filtered.is_empty() {
        return f64::NAN;
    }
    filtered.sort_by(|a, b| a.partial_cmp(b).expect("filtered values are finite"));
    let mid = filtered.len() / 2;
    if filtered.len() % 2 == 0 {
        (filtered[mid - 1] + filtered[mid]) / 2.0
    } else {
        filtered[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_empty() {
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn median_ignores_nan_and_inf() {
        let data = [1.0, f64::NAN, 3.0, f64::INFINITY, 2.0, f64::NEG_INFINITY];
        assert_eq!(median(&data), 2.0);
    }
}