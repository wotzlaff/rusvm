use crate::max::poly2;
use crate::status::Status;

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

    fn dual_loss(&self, i: usize, ai: f64) -> f64 {
        let wi = self.weight(i);
        let ya = self.y[i] * ai;
        wi * poly2::dual_max(ya / wi, self.params.smoothing) - self.shift * ya
    }
    fn d_dual_loss(&self, i: usize, ai: f64) -> f64 {
        let wi = self.weight(i);
        let yi = self.y[i];
        yi * (poly2::d_dual_max(yi * ai / wi, self.params.smoothing) - self.shift)
    }
    fn d2_dual_loss(&self, i: usize, ai: f64) -> f64 {
        let wi = self.weight(i);
        let yi = self.y[i];
        poly2::d2_dual_max(yi * ai / wi, self.params.smoothing) / wi
    }
    fn loss(&self, i: usize, ti: f64) -> f64 {
        self.weight(i) * poly2::max(self.shift - self.y[i] * ti, self.params.smoothing)
    }
    fn d_loss(&self, i: usize, ti: f64) -> f64 {
        -self.y[i]
            * self.weight(i)
            * poly2::d_max(self.shift - self.y[i] * ti, self.params.smoothing)
    }
    fn d2_loss(&self, i: usize, ti: f64) -> f64 {
        self.weight(i) * poly2::d2_max(self.shift - self.y[i] * ti, self.params.smoothing)
    }
}
