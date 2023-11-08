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

    /// Updates the verbosity level.
    pub fn with_verbose(mut self, verbose: usize) -> Self {
        self.verbose = verbose;
        self
    }

    /// Updates the termination tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Updates the time limit.
    pub fn with_time_limit(mut self, time_limit: f64) -> Self {
        self.time_limit = time_limit;
        self
    }

    /// Updates the maximum number of steps.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }
}
