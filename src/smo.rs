//! Sequential Minimal Optimization

mod params;
mod subproblem;
mod update;
mod ws;

pub use self::params::Params;

mod solve;
pub use solve::{solve, solve_with_status};
