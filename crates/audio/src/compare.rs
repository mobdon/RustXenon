//! Sound checks without ears: compare a written stream with the original.

/// Best normalised correlation (-1..=1) of `b` against `a` over shifts of
/// `b` by 0..=`max_lag` samples later (decoders add delay), and the shift
/// that gave it. All of `a` is compared: where `b` runs out it counts as
/// silence, so a cut-short `b` scores lower. Both silent gives 1.0; one
/// silent, or an empty `a`, gives 0.0.
pub fn correlation(a: &[f32], b: &[f32], max_lag: usize) -> (f64, usize) {
    if a.is_empty() {
        return (0.0, 0);
    }
    let ea = a.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
    let mut best = (f64::MIN, 0);
    for lag in 0..=max_lag {
        let y = b.get(lag..).unwrap_or(&[]);
        let y = &y[..y.len().min(a.len())];
        let eb = y.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
        let c = match (ea == 0.0, eb == 0.0) {
            (true, true) => 1.0,
            (true, _) | (_, true) => 0.0,
            _ => {
                a.iter()
                    .zip(y)
                    .map(|(&p, &q)| p as f64 * q as f64)
                    .sum::<f64>()
                    / (ea * eb).sqrt()
            }
        };
        if c > best.0 {
            best = (c, lag);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_delay() {
        let a: Vec<f32> = (0..2000)
            .map(|i| (i as f32 * 0.05).sin() * (i as f32 * 0.003).cos())
            .collect();
        let mut b = vec![0.0; 37];
        b.extend(a.iter().map(|v| v * 0.5));
        let (c, lag) = correlation(&a, &b, 100);
        assert_eq!(lag, 37);
        assert!(c > 0.9999, "{c}");
        let neg: Vec<f32> = a.iter().map(|v| -v).collect();
        assert!(correlation(&a, &neg, 0).0 < -0.9999);
        assert_eq!(correlation(&a, &[0.0; 2000], 0).0, 0.0);
        assert_eq!(correlation(&[], &a, 0).0, 0.0);
        assert_eq!(correlation(&a, &[], 5).0, 0.0);
        // Half of the stream missing must not score as a match.
        assert!(correlation(&a, &a[..1000], 0).0 < 0.8);
    }
}
