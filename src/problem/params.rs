pub struct Params {
    pub smoothing: f64,
    pub lambda: f64,
    pub max_asum: f64,
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
