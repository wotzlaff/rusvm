#[derive(Clone, Debug)]
pub enum StatusCode {
    Initialized,
    Optimal,
    MaxSteps,
    TimeLimit,
    Callback,
}

#[derive(Clone, Debug)]
pub struct Status {
    pub a: Vec<f64>,
    pub b: f64,
    pub c: f64,
    pub asum: f64,
    pub violation: f64,
    pub value: f64,
    pub ka: Vec<f64>,
    pub g: Vec<f64>,
    pub code: StatusCode,
    pub steps: usize,
    pub time: f64,
}

impl Status {
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
