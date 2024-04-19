//! Solve SVM training problems.
#![warn(missing_docs)]
#![feature(trait_upcasting)]

#[cfg(feature = "wasm")]
#[macro_use]
mod console;

pub mod kernel;
mod max;
mod predict;
pub mod problem;
pub mod smo;
pub mod smonewt;
pub use crate::predict::predict;

pub mod newton;
pub mod sensitivity;

mod status;
pub use crate::status::{Status, StatusCode};
mod time;
