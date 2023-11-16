/// LS-SVM problem
pub struct LSSVM<'a> {
    y: &'a [f64],
    /// Parameters of the training problem
    pub params: super::Params,
}

impl<'a> LSSVM<'a> {
    /// Creates a [`LSSVM`] struct.
    ///
    /// * `y`: slice of labels
    /// * `params`: struct of problem parameters
    pub fn new(y: &[f64], params: super::Params) -> LSSVM {
        LSSVM { y, params }
    }
}

impl super::base::ProblemBase for LSSVM<'_> {
    fn size(&self) -> usize {
        self.y.len()
    }
    fn params(&self) -> &super::Params {
        &self.params
    }
}

impl super::shrinking::ShrinkingBase for LSSVM<'_> {
    fn lb(&self, _i: usize) -> f64 {
        f64::NEG_INFINITY
    }
    fn ub(&self, _i: usize) -> f64 {
        f64::INFINITY
    }
}

impl super::base::LabelProblem for LSSVM<'_> {
    type T = f64;
    fn label(&self, i: usize) -> f64 {
        self.y[i]
    }
}

impl super::PrimalLabelProblem for LSSVM<'_> {
    fn label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        let di = ti - yi;
        0.5 * di * di
    }
    fn d_label_loss(&self, _i: usize, ti: f64, yi: f64) -> f64 {
        ti - yi
    }
    fn d2_label_loss(&self, _i: usize, _ti: f64, _yi: f64) -> f64 {
        1.0
    }
}

impl super::DualLabelProblem for LSSVM<'_> {
    fn label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        ai * (0.5 * ai - yi)
    }
    fn d_label_dloss(&self, _i: usize, ai: f64, yi: f64) -> f64 {
        ai - yi
    }
    fn d2_label_dloss(&self, _i: usize, _ai: f64, _yi: f64) -> f64 {
        1.0
    }
    fn is_quad(&self) -> bool {
        true
    }
}
