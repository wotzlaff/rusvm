pub mod params;
pub use params::Params;

pub mod classification;
pub use classification::Classification;
pub mod regression;
pub use regression::Regression;

use crate::kernel::Kernel;
use crate::status::Status;

pub trait Problem {
    fn grad(&self, status: &Status, i: usize) -> f64 {
        status.ka[i] + self.d_dual_loss(i, status.a[i])
    }
    fn quad(&self, status: &Status, i: usize) -> f64 {
        // TODO: get rid of lambda here!
        self.d2_dual_loss(i, status.a[i])
    }
    fn size(&self) -> usize;
    fn lb(&self, i: usize) -> f64;
    fn ub(&self, i: usize) -> f64;
    fn sign(&self, i: usize) -> f64;
    fn is_optimal(&self, status: &Status, tol: f64) -> bool;

    fn params(&self) -> &Params;
    fn lambda(&self) -> f64 {
        self.params().lambda
    }
    fn smoothing(&self) -> f64 {
        self.params().smoothing
    }
    fn max_asum(&self) -> f64 {
        self.params().max_asum
    }
    fn regularization(&self) -> f64 {
        self.params().regularization
    }

    fn is_shrunk(&self, status: &Status, active_set: &Vec<usize>) -> bool {
        active_set.len() < status.a.len()
    }

    fn has_max_asum(&self) -> bool {
        f64::is_finite(self.max_asum())
    }

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

    fn dual_loss(&self, i: usize, ai: f64) -> f64;
    fn d_dual_loss(&self, i: usize, ai: f64) -> f64;
    fn d2_dual_loss(&self, i: usize, ai: f64) -> f64;
    fn loss(&self, i: usize, ti: f64) -> f64;
    fn d_loss(&self, i: usize, ti: f64) -> f64;
    fn d2_loss(&self, i: usize, ti: f64) -> f64;

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
    fn unshrink(&self, kernel: &mut dyn Kernel, status: &mut Status, active_set: &mut Vec<usize>) {
        let lambda = self.params().lambda;
        let n = self.size();
        let new_active_set = (0..n).collect();
        kernel.set_active(&active_set, &new_active_set);
        *active_set = new_active_set;

        status.ka.fill(0.0);
        for (i, &ai) in status.a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            kernel.use_rows([i].to_vec(), &active_set, &mut |ki_vec: Vec<&[f64]>| {
                let ki = ki_vec[0];
                for k in 0..n {
                    status.ka[k] += ai / lambda * ki[k];
                }
            })
        }
    }
}
