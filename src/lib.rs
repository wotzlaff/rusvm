//! Solve SVM training problems.
#![feature(trait_upcasting)]
#![warn(missing_docs)]

#[cfg(feature = "wasm")]
#[macro_use]
mod console;

pub mod kernel;
mod max;
pub mod newton;
mod predict;
pub mod problem;
pub mod smo;
pub mod smonewt;
pub use crate::predict::predict;

pub mod sensitivity;

mod status;
pub use crate::status::{Status, StatusCode};
mod time;
