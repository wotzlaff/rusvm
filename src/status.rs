use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// Step not possible
    NoStepPossible,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

    /// Determine support vectors and reduce the training set
    pub fn find_support<T: Clone>(&self, data: &Vec<T>) -> (Status, Vec<T>) {
        let size_support: usize = self.a.iter().map(|&ai| if ai != 0.0 { 1 } else { 0 }).sum();
        let mut new_status = Status::new(size_support);
        let mut new_data = Vec::with_capacity(size_support);
        let mut i = 0;
        for (idx, (&ai, xi)) in self.a.iter().zip(data).enumerate() {
            if ai != 0.0 {
                new_data.push(xi.clone());
                new_status.a[i] = self.a[idx];
                new_status.ka[i] = self.ka[idx];
                new_status.g[i] = self.g[idx];
                i += 1;
            }
        }
        new_status.b = self.b;
        new_status.c = self.c;
        (new_status, new_data)
    }
}
