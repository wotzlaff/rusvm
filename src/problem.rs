pub mod classification;
use crate::kernel::Kernel;
use crate::status::Status;
pub use classification::Classification;

pub trait Problem {
    fn quad(&self, status: &Status, i: usize) -> f64;
    fn grad(&self, status: &Status, i: usize) -> f64;
    fn size(&self) -> usize;
    fn lb(&self, i: usize) -> f64;
    fn ub(&self, i: usize) -> f64;
    fn sign(&self, i: usize) -> f64;
    fn is_optimal(&self, status: &Status, tol: f64) -> bool;

    fn lambda(&self) -> f64;
    fn regularization(&self) -> f64;

    fn is_shrunk(&self, status: &Status, active_set: &Vec<usize>) -> bool {
        active_set.len() < status.a.len()
    }

    fn objective(&self, status: &Status) -> (f64, f64);

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
        let lambda = self.lambda();
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
