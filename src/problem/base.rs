//! Training problem base
use super::params::Params;

use crate::kernel::Kernel;
use crate::status::Status;

/// Base for the definition of a training problem
pub trait ProblemBase {
    /// Returns the size of the optimization problem (the number of variables).
    fn size(&self) -> usize;
    /// Returns the sign of the ith variable.
    fn sign(&self, _i: usize) -> f64 {
        0.0
    }

    /// Returns the parameters of the training problem.
    fn params(&self) -> &Params;
    /// Returns the regularization parameter lambda.
    fn lambda(&self) -> f64 {
        self.params().lambda
    }
    /// Returns the smoothing parameter of the max function.
    fn smoothing(&self) -> f64 {
        self.params().smoothing
    }
    /// Returns the bound on the 1-norm of the coefficient vector.
    fn regularization(&self) -> f64 {
        self.params().regularization
    }
    /// Returns the bound on the 1-norm of the coefficient vector.
    fn max_asum(&self) -> f64 {
        self.params().max_asum
    }
    /// Checks whether a bound on the 1-norm of the coefficient is set.
    fn has_max_asum(&self) -> bool {
        f64::is_finite(self.max_asum())
    }

    /// Checks for optimality.
    fn is_optimal(&self, status: &Status, tol: f64) -> bool {
        self.lambda() * status.violation < tol
    }

    /// Recomputes the product of kernel matrix and coefficient vector.
    fn recompute_kernel_product(
        &self,
        kernel: &mut dyn Kernel,
        status: &mut Status,
        active_set: &[usize],
    ) {
        let n = self.size();
        let lambda = self.params().lambda;
        status.ka.fill(0.0);
        for (i, &ai) in status.a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            kernel.use_rows([i].as_slice(), active_set, &mut |ki_vec: Vec<&[f64]>| {
                let ki = ki_vec[0];
                for k in 0..n {
                    status.ka[k] += ai / lambda * ki[k];
                }
            })
        }
    }
}

/// Training problem providing labels for each sample
pub trait LabelProblem: ProblemBase {
    /// Type of the labels
    type T;
    /// Gets the label of the ith sample.
    fn label(&self, i: usize) -> Self::T;
}
