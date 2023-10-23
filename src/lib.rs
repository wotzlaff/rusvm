mod kernel;
mod problem;
mod smo;
mod state;
pub use kernel::{GaussianKernel, Kernel};
pub use problem::{Classification, Problem};
pub use smo::{solve, SMOResult, Status};
pub use state::State;
