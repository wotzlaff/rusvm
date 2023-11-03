use crate::kernel::Kernel;
use crate::problem::Problem;
use std::time::Instant;

use super::status::{Status, StatusCode};

pub struct Params {
    pub tol: f64,
    pub max_steps: usize,
    pub verbose: usize,
    pub time_limit: f64,
}

impl Params {
    pub fn new() -> Self {
        Params {
            tol: 1e-4,
            max_steps: usize::MAX,
            verbose: 0,
            time_limit: f64::INFINITY,
        }
    }
}

pub fn solve(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let start = Instant::now();
    let n = problem.size();
    let mut status = Status::new(n);
    let mut step: usize = 0;
    let mut stop = false;
    loop {
        status.steps = step;
        let elapsed = start.elapsed().as_secs_f64();
        status.time = elapsed;
        if step >= params.max_steps {
            status.code = StatusCode::MaxSteps;
            stop = true;
        }
        if params.time_limit > 0.0 && elapsed >= params.time_limit {
            status.code = StatusCode::TimeLimit;
            stop = true;
        }

        if let Some(callback_fn) = callback {
            stop = callback_fn(&status);
            if stop {
                status.code = StatusCode::Callback;
            }
        };

        let optimal = problem.is_optimal(&status, params.tol);

        if params.verbose > 0 && (step % params.verbose == 0 || optimal) {
            println!(
                "{:10} {:10.2} {:10.6} {:10.6} {:8.3}",
                step, elapsed, status.violation, status.value, status.asum,
            )
        }

        if optimal {
            status.code = StatusCode::Optimal;
            stop = true;
        }

        if stop {
            break;
        }

        // TODO: implement something

        step += 1;
    }
    status
}
