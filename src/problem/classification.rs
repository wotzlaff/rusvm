use crate::max::{dual_smooth_max_2, smooth_max_2};
use crate::smo::status::Status;

pub struct Classification<'a> {
    y: &'a [f64],
    w: Option<&'a [f64]>,
    pub params: super::Params,
    pub shift: f64,
}

impl<'a> Classification<'a> {
    pub fn new(y: &[f64], params: super::Params) -> Classification {
        Classification {
            y,
            w: None,
            params,
            shift: 1.0,
        }
    }

    fn weight(&self, i: usize) -> f64 {
        match self.w {
            Some(w) => w[i],
            None => 1.0,
        }
    }

    pub fn with_weights(mut self, w: &'a [f64]) -> Self {
        self.w = Some(w);
        self
    }

    pub fn with_shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }
}

impl<'a> super::Problem for Classification<'a> {
    fn quad(&self, _status: &Status, i: usize) -> f64 {
        2.0 * self.params.smoothing * self.params.lambda / self.weight(i)
    }
    fn grad(&self, status: &Status, i: usize) -> f64 {
        status.ka[i] - self.shift * self.y[i]
            + self.params.smoothing
                * self.y[i]
                * (2.0 * self.y[i] * status.a[i] / self.weight(i) - 1.0)
    }
    fn size(&self) -> usize {
        self.y.len()
    }
    fn lb(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            0.0
        } else {
            -self.weight(i)
        }
    }
    fn ub(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            self.weight(i)
        } else {
            0.0
        }
    }
    fn sign(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn is_optimal(&self, status: &Status, tol: f64) -> bool {
        self.params.lambda * status.violation < tol
    }

    fn params(&self) -> &super::Params {
        &self.params
    }

    fn objective(&self, status: &Status) -> (f64, f64) {
        let mut reg = 0.0;
        let mut loss_primal = 0.0;
        let mut loss_dual = 0.0;
        for i in 0..self.size() {
            reg += status.ka[i] * status.a[i];
            let dec = status.ka[i] + status.b + self.sign(i) * status.c;
            let ya = self.y[i] * status.a[i];
            loss_primal +=
                self.weight(i) * smooth_max_2(self.shift - self.y[i] * dec, self.params.smoothing);
            loss_dual += self.weight(i)
                * dual_smooth_max_2(ya / self.weight(i), self.params.smoothing)
                - self.shift * ya;
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
