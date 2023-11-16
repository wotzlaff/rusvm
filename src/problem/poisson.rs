/// Poisson regression problem
pub struct Poisson<'a> {
    y: &'a [f64],
    /// Parameters of the training problem
    pub params: super::Params,
}

impl<'a> Poisson<'a> {
    /// Creates a [`Poisson`] struct.
    ///
    /// * `y`: slice of labels
    /// * `params`: struct of problem parameters
    pub fn new(y: &[f64], params: super::Params) -> Poisson {
        assert!(
            y.iter().all(|&yi| yi >= 0.0),
            "labels should be non-negative"
        );
        Poisson { y, params }
    }
}

impl super::base::ProblemBase for Poisson<'_> {
    fn size(&self) -> usize {
        self.y.len()
    }
    fn params(&self) -> &super::Params {
        &self.params
    }
}

impl super::shrinking::ShrinkingBase for Poisson<'_> {
    fn lb(&self, _i: usize) -> f64 {
        f64::NEG_INFINITY
    }
    fn ub(&self, i: usize) -> f64 {
        self.y[i]
    }
}

impl super::base::LabelProblem for Poisson<'_> {
    type T = f64;
    fn label(&self, i: usize) -> f64 {
        self.y[i]
    }
}

impl super::PrimalLabelProblem for Poisson<'_> {
    fn label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        ti.exp() - yi * ti
    }
    fn d_label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        ti.exp() - yi
    }
    fn d2_label_loss(&self, _i: usize, ti: f64, _yi: f64) -> f64 {
        ti.exp()
    }
}

impl super::DualLabelProblem for Poisson<'_> {
    fn label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        let yma = yi - ai;
        if yma == 0.0 {
            0.0
        } else {
            yma * (yma.ln() - 1.0)
        }
    }
    fn d_label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        let yma = yi - ai;
        -yma.ln()
    }
    fn d2_label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        let yma = yi - ai;
        1.0 / yma
    }
}
