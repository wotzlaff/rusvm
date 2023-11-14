/// Poisson regression problem
pub struct Poisson<'a> {
    y: &'a [f64],
    w: Option<&'a [f64]>,
    /// Parameters of the training problem
    pub params: super::Params,
}

impl<'a> Poisson<'a> {
    /// Creates a [`Poisson`] struct.
    ///
    /// * `y`: slice of labels (non-negative integers)
    /// * `params`: struct of problem parameters
    pub fn new(y: &[f64], params: super::Params) -> Poisson {
        Poisson { y, w: None, params }
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
}

impl<'a> super::Problem for Poisson<'a> {
    fn size(&self) -> usize {
        self.y.len()
    }
    fn lb(&self, _i: usize) -> f64 {
        f64::NEG_INFINITY
    }
    fn ub(&self, i: usize) -> f64 {
        self.y[i]
    }
    fn sign(&self, _i: usize) -> f64 {
        0.0
    }

    fn params(&self) -> &super::Params {
        &self.params
    }

    fn dual_loss(&self, i: usize, ai: f64) -> f64 {
        let wi = self.weight(i);
        let yma = self.y[i] - ai / wi;
        if yma == 0.0 {
            0.0
        } else {
            wi * yma * (yma.ln() - 1.0)
        }
    }
    fn d_dual_loss(&self, i: usize, ai: f64) -> f64 {
        let wi = self.weight(i);
        let yma = self.y[i] - ai / wi;
        -yma.ln()
    }
    fn d2_dual_loss(&self, i: usize, ai: f64) -> f64 {
        let wi = self.weight(i);
        let yma = self.y[i] - ai / wi;
        1.0 / (wi * yma)
    }
    fn loss(&self, i: usize, ti: f64) -> f64 {
        self.weight(i) * (ti.exp() - self.y[i] * ti)
    }
    fn d_loss(&self, i: usize, ti: f64) -> f64 {
        self.weight(i) * (ti.exp() - self.y[i])
    }
    fn d2_loss(&self, i: usize, ti: f64) -> f64 {
        self.weight(i) * ti.exp()
    }
}
