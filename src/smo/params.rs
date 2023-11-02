pub struct Params {
    pub tol: f64,
    pub max_steps: usize,
    pub verbose: usize,
    pub log_objective: bool,
    pub second_order: bool,
    pub shrinking_period: usize,
    pub shrinking_threshold: f64,
    pub time_limit: f64,
}

impl Params {
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
}
