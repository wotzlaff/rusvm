//! Solve SVM training problems.
#![feature(trait_upcasting)]
#![warn(missing_docs)]

pub mod kernel;
mod max;
pub mod newton;
pub mod problem;
pub mod smo;
pub mod smonewt;

mod status;
pub use crate::status::{Status, StatusCode};
