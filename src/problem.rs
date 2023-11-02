pub mod classification;
pub mod regression;
use crate::kernel::Kernel;
use crate::status::Status;
pub use classification::Classification;
pub use regression::Regression;

pub struct Params {
    pub smoothing: f64,
    pub lambda: f64,
    pub max_asum: f64,
    pub regularization: f64,
}

impl Params {
    const DEFAULT_SMOOTHING: f64 = 0.0;
    const DEFAULT_LAMBDA: f64 = 1.0;
    const DEFAULT_MAX_ASUM: f64 = f64::INFINITY;
    const DEFAULT_REGULARIZATION: f64 = 1e-12;

    pub fn new() -> Self {
        Params {
            smoothing: Self::DEFAULT_SMOOTHING,
            lambda: Self::DEFAULT_LAMBDA,
            max_asum: Self::DEFAULT_MAX_ASUM,
            regularization: Self::DEFAULT_REGULARIZATION,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    pub fn with_max_asum(mut self, max_asum: f64) -> Self {
        self.max_asum = max_asum;
        self
    }

    pub fn with_regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }
}

pub trait Problem {
    fn quad(&self, status: &Status, i: usize) -> f64;
    fn grad(&self, status: &Status, i: usize) -> f64;
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
