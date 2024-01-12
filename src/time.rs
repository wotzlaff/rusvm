#[cfg(not(feature = "wasm-time"))]
use std::time::Instant;
#[cfg(not(feature = "wasm-time"))]
pub fn now() -> Instant {
    Instant::now()
}
#[cfg(not(feature = "wasm-time"))]
pub fn until_now(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}
#[cfg(feature = "wasm-time")]
pub fn now() -> f64 {
    web_sys::window()
        .expect("should have a Window")
        .performance()
        .expect("should have a Performance")
        .now()
}
#[cfg(feature = "wasm-time")]
pub fn until_now(t: f64) -> f64 {
    (now() - t) / 1000.0
}
