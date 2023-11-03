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

    // status.dec = vec![0.0; n];
    status.h = vec![0.0; n];
    loop {
        // update steps and time
        status.steps = step;
        let elapsed = start.elapsed().as_secs_f64();
        status.time = elapsed;

        // handle step limit
        if step >= params.max_steps {
            status.code = StatusCode::MaxSteps;
            stop = true;
        }

        // handle time limit
        if params.time_limit > 0.0 && elapsed >= params.time_limit {
            status.code = StatusCode::TimeLimit;
            stop = true;
        }

        // handle callback
        if let Some(callback_fn) = callback {
            stop = callback_fn(&status);
            if stop {
                status.code = StatusCode::Callback;
            }
        };

        // check for optimality
        let optimal = problem.is_optimal(&status, params.tol);
        if optimal {
            status.code = StatusCode::Optimal;
            stop = true;
        }

        // handle progress output
        if params.verbose > 0 && (step % params.verbose == 0 || optimal) {
            println!(
                "{:10} {:10.2} {:10.6} {:10.6} {:8.3}",
                step, elapsed, status.violation, status.value, status.asum,
            )
        }

        // terminate
        if stop {
            break;
        }

        // compute decisions
        for i in 0..problem.size() {
            let ti = status.ka[i] + status.b + status.c * problem.sign(i);
            status.g[i] = problem.d_loss(i, ti);
            status.h[i] = problem.d2_loss(i, ti);
        }

        step += 1;
    }
    status
}
