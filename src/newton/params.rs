/// Parameters of Newton's method
#[derive(Debug)]
pub struct Params {
    /// Termination tolerance
    pub tol: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Frequency of logging (`0` for no logging)
    pub verbose: usize,
    /// Time limit (in seconds)
    pub time_limit: f64,
}

impl Params {
    /// Creates a new [`Params`] struct with default parameter values.
    pub fn new() -> Self {
        Params {
            tol: 1e-8,
            max_steps: usize::MAX,
            verbose: 0,
            time_limit: f64::INFINITY,
        }
    }
}
