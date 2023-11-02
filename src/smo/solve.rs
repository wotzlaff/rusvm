use crate::kernel::Kernel;
use crate::problem::Problem;
use std::time::Instant;

// pub mod status;
use super::status::{Status, StatusCode};
use super::update::update;
use super::ws::*;

pub fn solve(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    tol: f64,
    max_steps: usize,
    verbose: usize,
    log_objective: bool,
    second_order: bool,
    shrinking_period: usize,
    shrinking_threshold: f64,
    time_limit: f64,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let start = Instant::now();
    let n = problem.size();
    let mut active_set = (0..n).collect();

    let mut status = Status::new(n);
    let mut step: usize = 0;
    let mut stop = false;
    loop {
        status.steps = step;
        let elapsed = start.elapsed().as_secs_f64();
        status.time = elapsed;
        if step >= max_steps {
            status.code = StatusCode::MaxSteps;
            stop = true;
        }
        if time_limit > 0.0 && elapsed >= time_limit {
            status.code = StatusCode::TimeLimit;
            stop = true;
        }

        if let Some(callback_fn) = callback {
            stop = callback_fn(&status);
            if stop {
                status.code = StatusCode::Callback;
            }
        };

        if shrinking_period > 0 && step % shrinking_period == 0 {
            problem.shrink(kernel, &status, &mut active_set, shrinking_threshold);
        }

        let (idx_i0, idx_j1) = find_mvp(problem, &mut status, &active_set);
        let optimal = problem.is_optimal(&status, tol);

        if verbose > 0 && (step % verbose == 0 || optimal) {
            if log_objective {
                let (obj_primal, obj_dual) = problem.objective(&status);
                let gap = obj_primal + obj_dual;
                println!(
                    "{:10} {:10.2} {:10.6} {:10.6} {:10.6} {:10.6} {:10.6} {:8.3} {:8} / {}",
                    step,
                    elapsed,
                    status.violation,
                    gap,
                    obj_primal,
                    -obj_dual,
                    status.value,
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

        if optimal {
            if problem.is_shrunk(&status, &active_set) {
                problem.unshrink(kernel, &mut status, &mut active_set);
                continue;
            }
            status.code = StatusCode::Optimal;
            stop = true;
        }

        if stop {
            break;
        }

        let (idx_i, idx_j) = if second_order {
            let sign = if problem.has_max_asum() && status.asum == problem.max_asum() {
                problem.sign(active_set[idx_i0])
            } else {
                0.0
            };
            find_ws2(problem, kernel, idx_i0, idx_j1, &status, &active_set, sign)
        } else {
            (idx_i0, idx_j1)
        };
        update(problem, kernel, idx_i, idx_j, &mut status, &active_set);
        step += 1;
    }
    status
}
