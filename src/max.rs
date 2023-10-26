pub fn smooth_max_2(t: f64, s: f64) -> f64 {
    if t >= s {
        t
    } else if t <= -s {
        0.0
    } else {
        0.25 / s * (t + s) * (t + s)
    }
}

pub fn dual_smooth_max_2(a: f64, s: f64) -> f64 {
    s * a * (a - 1.0)
}
