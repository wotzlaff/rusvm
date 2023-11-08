#[derive(Debug)]
/// Parameters of the SMO method
pub struct Params {
    /// Termination tolerance
    pub tol: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Frequency of logging or `0` for no logging
    pub verbose: usize,
    /// Decides whether or not to (compute and) log (primal and dual) objective function values.
    /// This option only affects the logging, but potentially increases the rutime if `true`.
    pub log_objective: bool,
    /// Decides whether or not second-order information is used for the working set selection. Should be `true`.
    pub second_order: bool,
    /// Shrinking frequency (number of steps between application of shrinking)
    pub shrinking_period: usize,
    /// Threshold of shrinking decision:
    /// Larger values lead to more (but potentially wrong) shrinking.
    pub shrinking_threshold: f64,
    /// Time limit (in seconds)
    pub time_limit: f64,
}

impl Params {
    /// Creates a new [`Params`] struct with default parameter values.
    pub fn new() -> Self {
        Params {
            tol: 1e-4,
            max_steps: usize::MAX,
            verbose: 0,
            log_objective: false,
            second_order: true,
            shrinking_period: 0,
            shrinking_threshold: 1.0,
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

    /// Updates the shrinking frequency.
    pub fn with_shrinking_period(mut self, shrinking_period: usize) -> Self {
        self.shrinking_period = shrinking_period;
        self
    }
}
