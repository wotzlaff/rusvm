use std::time::Instant;
pub fn now() -> Instant {
    Instant::now()
}
pub fn until_now(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}
