use crate::max::{dual_smooth_max_2, smooth_max_2};
use crate::status::Status;

pub struct Regression<'a> {
    y: &'a [f64],
    w: Option<&'a [f64]>,
    pub params: super::Params,
    pub epsilon: f64,
}

impl<'a> Regression<'a> {
    pub fn new(y: &[f64], params: super::Params) -> Regression {
        Regression {
            y,
            w: None,
            params,
            epsilon: 1e-6,
        }
    }

    fn weight(&self, i: usize) -> f64 {
        match self.w {
            Some(w) => w[i % w.len()],
            None => 1.0,
        }
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn with_weights(mut self, w: &'a [f64]) -> Self {
        self.w = Some(w);
        self
    }
}

impl<'a> super::Problem for Regression<'a> {
    fn quad(&self, _status: &Status, i: usize) -> f64 {
        2.0 * self.params.smoothing * self.params.lambda / self.weight(i)
    }
    fn grad(&self, status: &Status, i: usize) -> f64 {
        let yi = self.y[i % self.y.len()];
        status.ka[i] - yi
            + self.sign(i)
                * (self.epsilon
                    + self.params.smoothing
                        * (2.0 * self.sign(i) * status.a[i] / self.weight(i) - 1.0))
    }
    fn size(&self) -> usize {
        2 * self.y.len()
    }
    fn lb(&self, i: usize) -> f64 {
        let n = self.y.len();
        if i < n {
            0.0
        } else {
            -self.weight(i - n)
        }
    }
    fn ub(&self, i: usize) -> f64 {
        let n = self.y.len();
        if i < n {
            self.weight(i)
        } else {
            0.0
        }
    }
    fn sign(&self, i: usize) -> f64 {
        if i < self.y.len() {
            1.0
        } else {
            -1.0
        }
    }

    fn is_optimal(&self, status: &Status, tol: f64) -> bool {
        self.params.lambda * status.violation < tol
    }

    fn lambda(&self) -> f64 {
        self.params.lambda
    }
    fn regularization(&self) -> f64 {
        1e-12
    }

    fn objective(&self, status: &Status) -> (f64, f64) {
        let mut reg = 0.0;
        let mut loss_primal = 0.0;
        let mut loss_dual = 0.0;
        for i in 0..self.size() {
            reg += status.ka[i] * status.a[i];
            let yi = self.y[i % self.y.len()];
            let wi = self.weight(i);
            let si = self.sign(i);
            let dec = status.ka[i] + status.b - si * status.c;
            let ya = yi * status.a[i];
            loss_primal += self.weight(i)
                * smooth_max_2(si * (dec - yi) - self.epsilon, self.params.smoothing);
            loss_dual += self.weight(i)
                * dual_smooth_max_2(status.a[i] / wi * si, self.params.smoothing)
                - ya
                + self.epsilon * si * status.a[i];
        }
        let asum_term = if self.params.max_asum < f64::INFINITY {
            self.params.max_asum * status.c
        } else {
            0.0
        };
        let obj_primal = 0.5 * reg + loss_primal + asum_term;
        let obj_dual = 0.5 * reg + loss_dual;
        (obj_primal, obj_dual)
    }
}
