use crate::max::poly2;

/// Basic SVM (hinge loss) classification problem
pub struct Classification<'a> {
    y: &'a [f64],
    w: Option<&'a [f64]>,
    /// Parameters of the training problem
    pub params: super::Params,
    /// Value of shift in the loss function: The typical value is `1`.
    pub shift: f64,
}

impl<'a> Classification<'a> {
    /// Creates a [`Classification`] struct.
    ///
    /// * `y`: slice of labels with values `-1.0` or `+1.0`
    /// * `params`: struct of problem parameters
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

    /// Sets the weights of the individual samples.
    pub fn with_weights(mut self, w: &'a [f64]) -> Self {
        self.w = Some(w);
        self
    }

    /// Sets the value of the shift in the loss function.
    pub fn with_shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }
}

impl<'a> super::Problem for Classification<'a> {
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
