//! Sequential Minimal Optimization

mod newton;
mod params;
mod update;
mod ws;

pub use self::params::Params;

mod solve;
pub use solve::{solve, solve_with_status};
