use crate::kernel::Kernel;
use crate::newton::status_extended::StatusExtended;
use crate::problem::PrimalProblem;

pub enum DirectionType {
    Gradient,
    Newton,
    NoStep,
}

pub fn gradient(
    problem: &dyn PrimalProblem,
    _kernel: &mut dyn Kernel,
    status_ext: &mut StatusExtended,
) {
    for i in 0..problem.size() {
        status_ext.dir.a[i] = status_ext.status.a[i] + status_ext.status.g[i];
    }
    status_ext.dir.b = status_ext.sums.g / problem.lambda();
}

#[cfg(feature = "lapack")]
mod lapack;
#[cfg(feature = "lapack")]
pub use lapack::newton_with_fallback;

#[cfg(not(feature = "lapack"))]
mod nolapack;
#[cfg(not(feature = "lapack"))]
pub use nolapack::newton_with_fallback;
