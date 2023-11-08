#[derive(Clone, Debug)]
/// Possible outcomes of an optimization routine
pub enum StatusCode {
    /// Optimization not started
    Initialized,
    /// Solution found (up to defined tolerance)
    Optimal,
    /// Maximum number of steps reached
    MaxSteps,
    /// Time limit reached
    TimeLimit,
    /// Stopped by the callbcak function
    Callback,
}

#[derive(Clone, Debug)]
/// A struct containing information about the current point and state of the optimization routine
pub struct Status {
    /// Vector of coefficients (typically called α in the literature)
    pub a: Vec<f64>,
    /// Value of offset (bias) of the decision function
    pub b: f64,
    /// Value of additional shift of the decision function depending on the monotonicity of the loss function (applied in ν-SVM approach)
    pub c: f64,
    /// 1-norm of the coefficient vector
    pub asum: f64,
    /// Violation of optimality conditions
    pub violation: f64,
    /// Objective function value
    pub value: f64,
    /// Helper vector containing product of kernel matrix with coeffient vector (scaled by λ⁻¹)
    pub ka: Vec<f64>,
    /// Helper vector containing values of first-order derivatives
    pub g: Vec<f64>,
    /// Current status
    pub code: StatusCode,
    /// Number of conducted steps
    pub steps: usize,
    /// Elapsed time (in seconds)
    pub time: f64,
}

impl Status {
    /// Create a [`Status`] struct with default initialization for `n` samples
    pub fn new(n: usize) -> Status {
        Status {
            a: vec![0.0; n],
            b: 0.0,
            c: 0.0,
            asum: 0.0,
            violation: f64::INFINITY,
            value: 0.0,
            ka: vec![0.0; n],
            g: vec![0.0; n],
            code: StatusCode::Initialized,
            steps: 0,
            time: 0.0,
        }
    }
}
