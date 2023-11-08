/// Common parameters of a training problem
pub struct Params {
    /// Extent of smoothing of the use max function
    pub smoothing: f64,
    /// Regularization parameter in training problem
    pub lambda: f64,
    /// Maximum 1-norm of coefficient vector for additional sparsity (comparable to parameter in ν-SVM)
    pub max_asum: f64,
    /// Regularization parameter for descent estimation
    pub regularization: f64,
}

impl Params {
    const DEFAULT_SMOOTHING: f64 = 0.0;
    const DEFAULT_LAMBDA: f64 = 1.0;
    const DEFAULT_MAX_ASUM: f64 = f64::INFINITY;
    const DEFAULT_REGULARIZATION: f64 = 1e-12;

    pub fn new() -> Self {
        Params {
            smoothing: Self::DEFAULT_SMOOTHING,
            lambda: Self::DEFAULT_LAMBDA,
            max_asum: Self::DEFAULT_MAX_ASUM,
            regularization: Self::DEFAULT_REGULARIZATION,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    pub fn with_max_asum(mut self, max_asum: f64) -> Self {
        self.max_asum = max_asum;
        self
    }

    pub fn with_regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }
}
