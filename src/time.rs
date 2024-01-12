#[cfg(feature = "wasm-time")]
mod wasm;
#[cfg(feature = "wasm-time")]
pub use wasm::{now, until_now};

#[cfg(not(feature = "wasm-time"))]
mod std;
#[cfg(not(feature = "wasm-time"))]
pub use std::{now, until_now};
