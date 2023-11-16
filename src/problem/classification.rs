use crate::max::poly2;

/// Basic SVM (hinge loss) classification problem
pub struct Classification<'a> {
    y: &'a [f64],
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
        assert!(
            y.iter().all(|&yi| yi == 1.0 || yi == -1.0),
            "labels should be -1 or +1"
        );
        Classification {
            y,
            params,
            shift: 1.0,
        }
    }
    /// Sets the value of the shift in the loss function.
    pub fn with_shift(mut self, shift: f64) -> Self {
        self.shift = shift;
        self
    }
}

impl super::base::ProblemBase for Classification<'_> {
    fn size(&self) -> usize {
        self.y.len()
    }
    fn sign(&self, i: usize) -> f64 {
        self.y[i]
    }
    fn params(&self) -> &super::Params {
        &self.params
    }
}

impl super::shrinking::ShrinkingBase for Classification<'_> {
    fn lb(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            0.0
        } else {
            -1.0
        }
    }
    fn ub(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

impl super::base::LabelProblem for Classification<'_> {
    type T = f64;
    fn label(&self, i: usize) -> f64 {
        self.y[i]
    }
}

impl super::PrimalLabelProblem for Classification<'_> {
    fn label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        poly2::max(self.shift - yi * ti, self.params.smoothing)
    }
    fn d_label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        -yi * poly2::d_max(self.shift - yi * ti, self.params.smoothing)
    }
    fn d2_label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        poly2::d2_max(self.shift - yi * ti, self.params.smoothing)
    }
}

impl super::DualLabelProblem for Classification<'_> {
    fn label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        let ya = yi * ai;
        poly2::dual_max(ya, self.params.smoothing) - self.shift * ya
    }
    fn d_label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        yi * (poly2::d_dual_max(yi * ai, self.params.smoothing) - self.shift)
    }
    fn d2_label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        poly2::d2_dual_max(yi * ai, self.params.smoothing)
    }
    fn is_quad(&self) -> bool {
        true
    }
}
