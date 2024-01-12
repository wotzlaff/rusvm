#[cfg(feature = "wasm")]
mod wasm;
#[cfg(feature = "wasm")]
pub use wasm::{now, until_now};

#[cfg(not(feature = "wasm"))]
mod std;
#[cfg(not(feature = "wasm"))]
pub use std::{now, until_now};
