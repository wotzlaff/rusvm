use crate::status::Status;

pub struct Classification<'a> {
    pub smoothing: f64,
    pub lambda: f64,
    pub shift: f64,
    y: &'a [f64],
    w: Option<&'a [f64]>,
}

impl<'a> Classification<'a> {
    pub fn new(y: &[f64], lambda: f64) -> Classification {
        Classification {
            smoothing: 0.0,
            lambda,
            shift: 1.0,
            y,
            w: None,
        }
    }

    fn weight(&self, i: usize) -> f64 {
        match self.w {
            Some(w) => w[i],
            None => 1.0,
        }
    }

    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    pub fn with_shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }

    pub fn with_weights(mut self, w: &'a [f64]) -> Self {
        self.w = Some(w);
        self
    }
}

impl<'a> super::Problem for Classification<'a> {
    fn quad(&self, _state: &Status, i: usize) -> f64 {
        2.0 * self.smoothing * self.lambda / self.weight(i)
    }
    fn grad(&self, state: &Status, i: usize) -> f64 {
        state.ka[i] - self.shift * self.y[i]
            + self.smoothing * self.y[i] * (2.0 * self.y[i] * state.a[i] / self.weight(i) - 1.0)
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

    fn is_optimal(&self, state: &Status, tol: f64) -> bool {
        self.lambda * state.violation < tol
    }

    fn lambda(&self) -> f64 {
        self.lambda
    }
    fn regularization(&self) -> f64 {
        1e-12
    }
}
