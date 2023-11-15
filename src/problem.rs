//! Problems

pub mod base;
mod shrinking;

mod params;
pub use params::Params;

mod dual;
pub use dual::{DualLabelProblem, DualProblem};
mod primal;
pub use primal::{PrimalLabelProblem, PrimalProblem};

mod classification;
pub use classification::Classification;
mod regression;
pub use regression::Regression;
mod lssvm;
pub use lssvm::LSSVM;
mod poisson;
pub use poisson::Poisson;

pub trait Problem: PrimalProblem + DualProblem {}
impl<P> Problem for P where P: PrimalProblem + DualProblem {}
