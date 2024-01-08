//! Dual training problem
use super::base::ProblemBase;
use crate::kernel::Kernel;
use crate::status::Status;

/// Base for training problem definition
pub trait ShrinkingBase: ProblemBase {
    /// Returns the lower bound of the ith variable.
    fn lb(&self, i: usize) -> f64;
    /// Returns the upper bound of the ith variable.
    fn ub(&self, i: usize) -> f64;

    /// Checks whether the problem is shrunk.
    fn is_shrunk(&self, status: &Status, active_set: &Vec<usize>) -> bool {
        active_set.len() < status.a.len()
    }

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
                gkb_sqr <= threshold * status.opt_status.violation
                    || !(status.a[k] == self.ub(k) && gkb < 0.0
                        || status.a[k] == self.lb(k) && gkb > 0.0)
            })
            .collect();
        kernel.restrict_active(&active_set, &new_active_set);
        *active_set = new_active_set;
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
