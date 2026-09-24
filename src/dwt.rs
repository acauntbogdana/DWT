//! Last-out-first-in DWT model.
//!
//! Folding-stack с wiping-out памятью.
//! 1. Извлекаем чередующиеся локальные экстремумы (с порогом амплитуды).
//! 2. Свёртываем их по правилу wiping-out в LIFO-стек выживших разворотов.
//!
//! Инвариант: максимумы строго убывают, минимумы строго возрастают.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Max,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extreme {
    pub t: usize,
    pub x: f64,
    pub kind: Kind,
}

impl Extreme {
    pub fn new(t: usize, x: f64, kind: Kind) -> Self {
        Self { t, x, kind }
    }

    /// `self` доминирует над `other` согда они одного типа и `self` строго экстремальнее.
    pub fn dominates(&self, other: &Extreme) -> bool {
        if self.kind != other.kind {
            return false;
        }
        match self.kind {
            Kind::Max => self.x > other.x,
            Kind::Min => self.x < other.x,
        }
    }
}

/// Извлечение чередующихся локальных экстремумов с порогом амплитуды δ.
pub fn local_extrema(prices: &[f64], delta_rel: f64) -> Vec<Extreme> {
    let n = prices.len();
    if n < 3 {
        return Vec::new();
    }
    let mut signs: Vec<i8> = Vec::with_capacity(n - 1);
    for w in prices.windows(2) {
        let d = w[1] - w[0];
        signs.push(if d > 0.0 { 1 } else if d < 0.0 { -1 } else { 0 });
    }
    let mut raw: Vec<Extreme> = Vec::new();
    for i in 1..signs.len() {
        let prev = signs[i - 1];
        let cur = signs[i];
        if prev > 0 && cur < 0 {
            raw.push(Extreme::new(i, prices[i], Kind::Max));
        } else if prev < 0 && cur > 0 {
            raw.push(Extreme::new(i, prices[i], Kind::Min));
        }
    }
    let mut out: Vec<Extreme> = Vec::new();
    for e in raw {
        match out.last() {
            None => out.push(e),
            Some(last) => {
                let denom = last.x.abs().max(1e-12);
                if (e.x - last.x).abs() / denom > delta_rel {
                    out.push(e);
                }
            }
        }
    }
    out
}

/// Свёртка последовательности экстремумов через правило wiping-out.
pub fn folding_stack(extrema: &[Extreme]) -> Vec<Extreme> {
    let mut stack: Vec<Extreme> = Vec::new();
    for &e in extrema {
        push_into_stack(&mut stack, e);
    }
    stack
}

/// Онлайн-обновление стека новым экстремумом.
///
/// Возвращает `true`, если при обработке произошло хотя бы одно стирание
/// (wiping-out), то есть имела место смена режима.
pub fn push_into_stack(stack: &mut Vec<Extreme>, e: Extreme) -> bool {
    let mut wiped = false;
    loop {
        let len = stack.len();
        if len == 0 {
            break;
        }
        let top = stack[len - 1];
        if top.kind == e.kind {
            if e.dominates(&top) {
                stack.pop();
                wiped = true;
            } else {
                return wiped;
            }
        } else if len >= 2 {
            let below = stack[len - 2];
            if e.dominates(&below) {
                stack.pop();
                stack.pop();
                wiped = true;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    stack.push(e);
    wiped
}

/// Полный pipeline: цены → экстремумы → folding stack.
pub fn skeleton(prices: &[f64], delta_rel: f64) -> Vec<Extreme> {
    folding_stack(&local_extrema(prices, delta_rel))
}

/// Состояние стека в момент времени `t`.
#[derive(Debug, Clone, Copy)]
pub struct DwtState {
    pub depth: usize,
    pub top_kind: Option<Kind>,
    pub top_price: f64,
    pub top2_price: f64,
    pub dist_to_top: f64,
    /// Средняя амплитуда между соседними выжившими разворотами.
    pub mean_amplitude: f64,
    /// RMS-амплитуда между соседними выжившими разворотами.
    pub rms_amplitude: f64,
    /// Среднее число баров между соседними выжившими разворотами.
    pub mean_gap_bars: f64,
}

impl DwtState {
    pub fn from_stack(stack: &[Extreme], current_price: f64) -> Self {
        let depth = stack.len();
        let top_kind = stack.last().map(|e| e.kind);
        let top_price = stack.last().map(|e| e.x).unwrap_or(f64::NAN);
        let top2_price = if stack.len() >= 2 {
            stack[stack.len() - 2].x
        } else {
            f64::NAN
        };
        let dist_to_top = if top_price.is_finite() {
            (current_price - top_price).abs()
        } else {
            f64::NAN
        };
        let (mean_amplitude, rms_amplitude, mean_gap_bars) = if stack.len() >= 2 {
            let mut s = 0.0;
            let mut s2 = 0.0;
            let mut gap_sum = 0.0;
            let mut k = 0usize;
            for w in stack.windows(2) {
                // Лог-амплитуда между соседними выжившими разворотами.
                // Цены > 0 гарантированы валидацией в analysis_core.
                let d = (w[1].x / w[0].x).ln().abs();
                s += d;
                s2 += d * d;
                // Расстояние в барах между соседними выжившими разворотами.
                // t — индекс бара в OnlineExtrema, поэтому разница корректна.
                gap_sum += (w[1].t as f64 - w[0].t as f64).max(1.0);
                k += 1;
            }
            if k > 0 {
                (s / k as f64, (s2 / k as f64).sqrt(), gap_sum / k as f64)
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            }
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };
        Self {
            depth,
            top_kind,
            top_price,
            top2_price,
            dist_to_top,
            mean_amplitude,
            rms_amplitude,
            mean_gap_bars,
        }
    }
}

/// Онлайн-трекер локальных экстремумов.
pub struct OnlineExtrema {
    prev_sign: i8,
    last_price: f64,
    pub stack: Vec<Extreme>,
    delta_rel: f64,
    t: usize,
    started: bool,
}

impl OnlineExtrema {
    pub fn new(delta_rel: f64) -> Self {
        Self {
            prev_sign: 0,
            last_price: f64::NAN,
            stack: Vec::new(),
            delta_rel,
            t: 0,
            started: false,
        }
    }

    /// Подать очередную цену.
    /// Возвращает (текущий стек, признак wiping-out на этом баре).
    pub fn push(&mut self, price: f64) -> (&[Extreme], bool) {
        if !self.started {
            self.last_price = price;
            self.started = true;
            self.t = 0;
            return (&self.stack, false);
        }
        self.t += 1;
        let d = price - self.last_price;
        let sign: i8 = if d > 0.0 {
            1
        } else if d < 0.0 {
            -1
        } else {
            0
        };
        let mut wiped = false;
        if sign != 0 {
            if self.prev_sign > 0 && sign < 0 {
                let e = Extreme::new(self.t - 1, self.last_price, Kind::Max);
                wiped = self.maybe_accept(e);
            } else if self.prev_sign < 0 && sign > 0 {
                let e = Extreme::new(self.t - 1, self.last_price, Kind::Min);
                wiped = self.maybe_accept(e);
            }
            self.prev_sign = sign;
        }
        self.last_price = price;
        (&self.stack, wiped)
    }

    fn maybe_accept(&mut self, e: Extreme) -> bool {
        let accept = match self.stack.last() {
            None => true,
            Some(last) if last.kind == e.kind => true,
            Some(last) => {
                let denom = last.x.abs().max(1e-12);
                (e.x - last.x).abs() / denom > self.delta_rel
            }
        };
        if accept {
            push_into_stack(&mut self.stack, e)
        } else {
            false
        }
    }

    pub fn state(&self, current_price: f64) -> DwtState {
        DwtState::from_stack(&self.stack, current_price)
    }
}