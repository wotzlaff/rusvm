//! Problems

mod params;
pub use params::Params;

mod classification;
pub use classification::Classification;
mod regression;
pub use regression::Regression;

use crate::kernel::Kernel;
use crate::status::Status;

/// Base for training problem definition
pub trait Problem {
    /// Computes the ith component gradient of the dual objective function.
    fn grad(&self, status: &Status, i: usize) -> f64 {
        status.ka[i] + self.d_dual_loss(i, status.a[i])
    }
    /// Computes the second derivative of the ith loss function.
    fn quad(&self, status: &Status, i: usize) -> f64 {
        self.d2_dual_loss(i, status.a[i])
    }

    /// Returns the size of the optimization problem (the number of variables).
    fn size(&self) -> usize;

    /// Returns the lower bound of the ith variable.
    fn lb(&self, i: usize) -> f64;
    /// Returns the upper bound of the ith variable.
    fn ub(&self, i: usize) -> f64;
    /// Returns the sign of the ith variable.
    fn sign(&self, i: usize) -> f64;

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
    fn max_asum(&self) -> f64 {
        self.params().max_asum
    }
    /// Returns the regularization parameter used for descent estimation.
    fn regularization(&self) -> f64 {
        self.params().regularization
    }

    /// Checks for optimality.
    fn is_optimal(&self, status: &Status, tol: f64) -> bool {
        self.lambda() * status.violation < tol
    }
    /// Checks whether the problem is shrunk.
    fn is_shrunk(&self, status: &Status, active_set: &Vec<usize>) -> bool {
        active_set.len() < status.a.len()
    }
    /// Checks whether a bound on the 1-norm of the coefficient is set.
    fn has_max_asum(&self) -> bool {
        f64::is_finite(self.max_asum())
    }

    /// Computes the primal and dual objective function values.
    fn objective(&self, status: &Status) -> (f64, f64) {
        let mut reg = 0.0;
        let mut loss_primal = 0.0;
        let mut loss_dual = 0.0;

        for i in 0..self.size() {
            // compute regularization
            reg += status.ka[i] * status.a[i];
            // compute primal loss
            let ti = status.ka[i] + status.b + self.sign(i) * status.c;
            loss_primal += self.loss(i, ti);
            // compute dual loss
            loss_dual += self.dual_loss(i, status.a[i]);
        }

        let asum_term = if self.max_asum() < f64::INFINITY {
            self.max_asum() * status.c
        } else {
            0.0
        };

        let obj_primal = 0.5 * reg + loss_primal + asum_term;
        let obj_dual = 0.5 * reg + loss_dual;
        (obj_primal, obj_dual)
    }

    /// Computes the ith loss function.
    fn loss(&self, i: usize, ti: f64) -> f64;
    /// Computes the first derivative of the ith loss function.
    fn d_loss(&self, i: usize, ti: f64) -> f64;
    /// Computes the second derivative of the ith loss function.
    fn d2_loss(&self, i: usize, ti: f64) -> f64;

    /// Computes the ith dual loss function.
    fn dual_loss(&self, i: usize, ai: f64) -> f64;
    /// Computes the first derivative of the ith dual loss function.
    fn d_dual_loss(&self, i: usize, ai: f64) -> f64;
    /// Computes the second derivative of the ith dual loss function.
    fn d2_dual_loss(&self, i: usize, ai: f64) -> f64;

    /// Conducts the shrinking procedure.
    fn shrink(
        &self,
        kernel: &mut dyn Kernel,
        status: &Status,
        active_set: &mut Vec<usize>,
        threshold: f64,
    ) {
        let new_active_set = active_set
            .to_vec()
            .into_iter()
            .filter(|&k| {
                let gkb = status.g[k] + status.b + status.c * self.sign(k);
                let gkb_sqr = gkb * gkb;
                gkb_sqr <= threshold * status.violation
                    || !(status.a[k] == self.ub(k) && gkb < 0.0
                        || status.a[k] == self.lb(k) && gkb > 0.0)
            })
            .collect();
        kernel.restrict_active(&active_set, &new_active_set);
        *active_set = new_active_set;
    }

    /// Recompute the product of kernel matrix and coefficient vector
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

    /// Revokes the shrinking procedure.
    fn unshrink(&self, kernel: &mut dyn Kernel, status: &mut Status, active_set: &mut Vec<usize>) {
        if !self.is_shrunk(status, active_set) {
            return;
        }
        let n = self.size();
        let new_active_set = (0..n).collect();
        kernel.set_active(&active_set, &new_active_set);
        *active_set = new_active_set;
        self.recompute_kernel_product(kernel, status, &active_set);
    }
}
