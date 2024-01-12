pub fn now() -> f64 {
    web_sys::window()
        .expect("should have a Window")
        .performance()
        .expect("should have a Performance")
        .now()
}
pub fn until_now(t: f64) -> f64 {
    (now() - t) / 1000.0
}
