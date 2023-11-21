use crate::kernel::Kernel;
use crate::problem::DualProblem;
use crate::status::{Status, StatusCode};
use std::time::Instant;

use super::update::update;
use super::ws::*;
use super::Params;

/// Uses the SMO method to solve the given training problem starting from the default initial point.
pub fn solve(
    problem: &dyn DualProblem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let n = problem.size();
    let mut status = Status::new(n);
    for k in 0..n {
        status.value -= problem.dloss(k, 0.0);
    }
    solve_with_status(status, problem, kernel, params, callback)
}

/// Uses the SMO method to solve the given training problem starting from a particular [`Status`].
pub fn solve_with_status(
    status: Status,
    problem: &dyn DualProblem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let mut status = status;
    let start = Instant::now();
    let n = problem.size();
    let mut active_set = (0..n).collect();

    let mut step: usize = 0;
    let mut stop = false;
    let mut last_ij = (0, 0);

    if params.verbose > 0 {
        if params.log_objective {
            println!(
                "{:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} / {}",
                "step", "time", "violation", "obj(inc)", "obj(comp)", "|a|", "|active|", "size",
            )
        } else {
            println!(
                "{:>10} {:>10} {:>10} {:>10} {:>8} {:>8} / {}",
                "step", "time", "violation", "obj(inc)", "|a|", "|active|", "size",
            )
        }
    }

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
            if callback_fn(&status) {
                status.code = StatusCode::Callback;
                stop = true;
            }
        };

        // handle shrinking
        if params.shrinking_period > 0 && step % params.shrinking_period == 0 {
            problem.shrink(kernel, &status, &mut active_set, params.shrinking_threshold);
        }

        // check for optimality
        let (idx_i0, idx_j1) = find_mvp(problem, &mut status, &active_set);
        let optimal = problem.is_optimal(&status, params.tol);
        if optimal {
            status.code = StatusCode::Optimal;
            stop = true;
        }

        // handle progress output
        if params.verbose > 0 && (step % params.verbose == 0 || optimal) {
            if params.log_objective {
                let obj = problem.objective(&status);
                println!(
                    "{:10} {:10.2} {:10.6} {:10.6} {:10.6} {:8.3} {:8} / {}",
                    step,
                    elapsed,
                    status.violation,
                    status.value,
                    -obj,
                    status.asum,
                    active_set.len(),
                    problem.size()
                )
            } else {
                println!(
                    "{:10} {:10.2} {:10.6} {:10.6} {:8.3} {:8} / {}",
                    step,
                    elapsed,
                    status.violation,
                    status.value,
                    status.asum,
                    active_set.len(),
                    problem.size()
                )
            }
        }

        // unshrink if necessary
        if optimal {
            if problem.is_shrunk(&status, &active_set) {
                problem.unshrink(kernel, &mut status, &mut active_set);
                continue;
            }
        }

        // terminate
        if stop {
            break;
        }

        // determine working set
        last_ij = if params.second_order {
            let sign = if problem.has_max_asum() && status.asum == problem.max_asum() {
                problem.sign(active_set[idx_i0])
            } else {
                0.0
            };
            let (idx_i2, idx_j2) =
                find_ws2(problem, kernel, idx_i0, idx_j1, &status, &active_set, sign);
            if (idx_i2, idx_j2) != last_ij {
                (idx_i2, idx_j2)
            } else {
                (idx_i0, idx_j1)
            }
        } else {
            (idx_i0, idx_j1)
        };

        let (idx_i, idx_j) = last_ij;

        // update selected variables
        update(problem, kernel, idx_i, idx_j, &mut status, &active_set);
        step += 1;
    }
    status
}
