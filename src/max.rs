pub mod poly2 {
    pub fn max(t: f64, s: f64) -> f64 {
        if t >= s {
            t
        } else if t <= -s {
            0.0
        } else {
            0.25 / s * (t + s) * (t + s)
        }
    }
    pub fn dual_max(a: f64, s: f64) -> f64 {
        s * a * (a - 1.0)
    }
}
