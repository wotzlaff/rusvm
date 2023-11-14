/// LS-SVM problem
pub struct LSSVM<'a> {
    y: &'a [f64],
    w: Option<&'a [f64]>,
    /// Parameters of the training problem
    pub params: super::Params,
}

impl<'a> LSSVM<'a> {
    /// Creates a [`LSSVM`] struct.
    ///
    /// * `y`: slice of labels
    /// * `params`: struct of problem parameters
    pub fn new(y: &[f64], params: super::Params) -> LSSVM {
        LSSVM { y, w: None, params }
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

impl<'a> super::Problem for LSSVM<'a> {
    fn size(&self) -> usize {
        self.y.len()
    }
    fn lb(&self, _i: usize) -> f64 {
        f64::NEG_INFINITY
    }
    fn ub(&self, _i: usize) -> f64 {
        f64::INFINITY
    }
    fn sign(&self, _i: usize) -> f64 {
        0.0
    }

    fn params(&self) -> &super::Params {
        &self.params
    }

    fn dual_loss(&self, i: usize, ai: f64) -> f64 {
        -ai * (self.y[i] - 0.5 * ai / self.weight(i))
    }
    fn d_dual_loss(&self, i: usize, ai: f64) -> f64 {
        ai / self.weight(i) - self.y[i]
    }
    fn d2_dual_loss(&self, i: usize, _ai: f64) -> f64 {
        1.0 / self.weight(i)
    }
    fn loss(&self, i: usize, ti: f64) -> f64 {
        let di = ti - self.y[i];
        0.5 * self.weight(i) * di * di
    }
    fn d_loss(&self, i: usize, ti: f64) -> f64 {
        let di = ti - self.y[i];
        self.weight(i) * di
    }
    fn d2_loss(&self, _i: usize, _ti: f64) -> f64 {
        1.0
    }
}
