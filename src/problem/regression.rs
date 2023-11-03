use crate::max::poly2;
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

    fn params(&self) -> &super::Params {
        &self.params
    }

    fn dual_loss(&self, i: usize, ai: f64) -> f64 {
        let si = self.sign(i);
        let yi = self.y[i % self.y.len()];
        let wi = self.weight(i);
        wi * poly2::dual_max(ai / wi * si, self.params.smoothing) - yi * ai + self.epsilon * si * ai
    }
    fn d_dual_loss(&self, i: usize, ai: f64) -> f64 {
        let si = self.sign(i);
        let yi = self.y[i % self.y.len()];
        let wi: f64 = self.weight(i);
        si * poly2::d_dual_max(ai / wi * si, self.params.smoothing) - yi + self.epsilon * si
    }
    fn d2_dual_loss(&self, i: usize, ai: f64) -> f64 {
        let si = self.sign(i);
        let wi: f64 = self.weight(i);
        poly2::d2_dual_max(ai / wi * si, self.params.smoothing) / wi
    }
    fn loss(&self, i: usize, ti: f64) -> f64 {
        let yi = self.y[i % self.y.len()];
        let si: f64 = self.sign(i);
        self.weight(i) * poly2::max(si * (yi - ti) - self.epsilon, self.params.smoothing)
    }
    fn d_loss(&self, i: usize, ti: f64) -> f64 {
        let yi = self.y[i % self.y.len()];
        let si = self.sign(i);
        -si * self.weight(i) * poly2::d_max(si * (yi - ti) - self.epsilon, self.params.smoothing)
    }
    fn d2_loss(&self, i: usize, ti: f64) -> f64 {
        let yi = self.y[i % self.y.len()];
        let si = self.sign(i);
        self.weight(i) * poly2::d2_max(si * (yi - ti) - self.epsilon, self.params.smoothing)
    }
}
