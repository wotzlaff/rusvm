//! Combination of SMO and Newton's method
use crate::kernel::Kernel;
use crate::problem::Problem;
use crate::Status;

use crate::newton;
use crate::smo;

/// Parameters of the combined method
pub struct Params {
    /// Parameters of the SMO method
    pub smo: smo::Params,
    /// Parameters of Newton's method
    pub newton: newton::Params,
}

impl Params {
    /// Creates a new [`Params`] struct with default parameter values.
    pub fn new() -> Params {
        Params {
            smo: smo::Params::new(),
            newton: newton::Params::new(),
        }
    }
}

/// Uses a combination of SMO and Newton's method to solve the given training problem starting from the default initial point.
pub fn solve(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback_smo: Option<&dyn Fn(&Status) -> bool>,
    callback_newton: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let n = problem.size();
    let mut status = smo::solve(problem, kernel, &params.smo, callback_smo);
    let full_set = (0..n).collect();
    kernel.set_active(&vec![], &full_set);
    problem.recompute_kernel_product(kernel, &mut status, &full_set);
    newton::solve_with_status(status, problem, kernel, &params.newton, callback_newton)
}
