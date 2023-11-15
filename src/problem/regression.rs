use super::base::ProblemBase;
use crate::max::poly2;

/// Basic SVM (ε-insensitive) regression problem
pub struct Regression<'a> {
    y: &'a [f64],
    /// Parameters of the training problem
    pub params: super::Params,
    /// Value of the parameter ε in the loss function: The default value is `1e-6` (to prevent degeneration).
    pub epsilon: f64,
}

impl<'a> Regression<'a> {
    /// Creates a [`Regression`] struct.
    ///
    /// * `y`: slice of labels with real values
    /// * `params`: struct of problem parameters
    pub fn new(y: &[f64], params: super::Params) -> Regression {
        Regression {
            y,
            params,
            epsilon: 1e-6,
        }
    }

    /// Sets the parameter ε in the loss function.
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl super::base::ProblemBase for Regression<'_> {
    fn size(&self) -> usize {
        2 * self.y.len()
    }
    fn sign(&self, i: usize) -> f64 {
        if i < self.y.len() {
            1.0
        } else {
            -1.0
        }
    }

    fn params(&self) -> &super::Params {
        &self.params
    }
}

impl super::shrinking::ShrinkingBase for Regression<'_> {
    fn lb(&self, i: usize) -> f64 {
        let n = self.y.len();
        if i < n {
            0.0
        } else {
            -1.0
        }
    }
    fn ub(&self, i: usize) -> f64 {
        let n = self.y.len();
        if i < n {
            1.0
        } else {
            0.0
        }
    }
}

impl super::base::LabelProblem for Regression<'_> {
    type T = f64;
    fn label(&self, i: usize) -> f64 {
        self.y[i % self.y.len()]
    }
}

impl super::PrimalLabelProblem for Regression<'_> {
    fn label_loss(&self, i: usize, ti: f64, yi: f64) -> f64 {
        let si: f64 = self.sign(i);
        poly2::max(si * (yi - ti) - self.epsilon, self.params.smoothing)
    }
    fn d_label_loss(&self, i: usize, ti: f64, yi: f64) -> f64 {
        let si = self.sign(i);
        -si * poly2::d_max(si * (yi - ti) - self.epsilon, self.params.smoothing)
    }
    fn d2_label_loss(&self, i: usize, ti: f64, yi: f64) -> f64 {
        let si = self.sign(i);
        poly2::d2_max(si * (yi - ti) - self.epsilon, self.params.smoothing)
    }
}

impl super::DualLabelProblem for Regression<'_> {
    fn label_dloss(&self, i: usize, ai: f64, yi: f64) -> f64 {
        let si = self.sign(i);
        poly2::dual_max(ai * si, self.params.smoothing) - yi * ai + self.epsilon * si * ai
    }
    fn d_label_dloss(&self, i: usize, ai: f64, yi: f64) -> f64 {
        let si = self.sign(i);
        si * poly2::d_dual_max(ai * si, self.params.smoothing) - yi + self.epsilon * si
    }
    fn d2_label_dloss(&self, i: usize, ai: f64, _yi: f64) -> f64 {
        let si = self.sign(i);
        poly2::d2_dual_max(ai * si, self.params.smoothing)
    }
}
