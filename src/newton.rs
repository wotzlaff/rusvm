//! Newton's Method

mod params;
pub use params::Params;
mod solve;
pub use solve::{solve, solve_with_status};
