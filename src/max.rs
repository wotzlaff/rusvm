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
    pub fn d_max(t: f64, s: f64) -> f64 {
        if t >= s {
            1.0
        } else if t <= -s {
            0.0
        } else {
            0.5 / s * (t + s)
        }
    }
    pub fn d2_max(t: f64, s: f64) -> f64 {
        if t >= s {
            0.0
        } else if t <= -s {
            0.0
        } else {
            0.5 / s
        }
    }
    pub fn dual_max(a: f64, s: f64) -> f64 {
        s * a * (a - 1.0)
    }
}
