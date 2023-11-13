use std::time::Instant;

use super::params::Params;
use super::status_extended::{ActiveSet, Direction, StatusExtended, Sums};
use crate::kernel::Kernel;
use crate::newton::direction::DirectionType;
use crate::problem::Problem;
use crate::status::{Status, StatusCode};

/// Uses a version of Newton's method to solve the given training problem starting from the default initial point.
pub fn solve(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let n = problem.size();
    let status = Status::new(n);
    solve_with_status(status, problem, kernel, params, callback)
}

fn compute_decisions(problem: &dyn Problem, status_ext: &mut StatusExtended) {
    let mut active = ActiveSet::new(problem.size());
    let mut sums = Sums::new();
    let mut violation = 0.0;
    for i in 0..problem.size() {
        let ai = status_ext.status.a[i];
        let si = problem.sign(i);
        sums.a += ai;
        sums.sa += si * ai;
        let ti = status_ext.status.ka[i] + status_ext.status.b + status_ext.status.c * si;
        let gi = problem.d_loss(i, ti);
        status_ext.status.g[i] = gi;
        sums.g += gi;
        sums.sg += gi * si;
        violation += (ai + gi).abs();
        let hi = problem.d2_loss(i, ti);
        status_ext.h[i] = hi;
        if hi == 0.0 {
            let dai = ai + gi;
            status_ext.dir.a[i] = dai;
            if dai != 0.0 {
                active.zeros.push(i);
                sums.da_zeros += dai;
            }
        } else {
            active.positive.push(i);
        }
    }
    violation += sums.a.abs();
    if problem.has_max_asum() {
        violation += (sums.sa - problem.max_asum()).abs();
    }
    status_ext.status.violation = violation;
    status_ext.status.asum = sums.sa;
    status_ext.active = active;
    status_ext.sums = sums;
}

fn update_status(
    problem: &dyn Problem,
    kernel: &dyn Kernel,
    status_ext: &mut StatusExtended,
    stepsize: f64,
) -> (f64, Status) {
    let sums = &status_ext.sums;
    let dir = &status_ext.dir;
    let status = &status_ext.status;
    let mut status_next = status.clone();
    let mut pred_desc = sums.g * dir.b;
    if problem.has_max_asum() {
        pred_desc += dir.c * (sums.sg + problem.max_asum());
    }
    status_next.b -= stepsize * dir.b;
    status_next.c -= stepsize * dir.c;
    for i in 0..problem.size() {
        if dir.a[i] == 0.0 {
            continue;
        }
        status_next.a[i] -= stepsize * dir.a[i];
        kernel.compute_row(
            i,
            &mut status_ext.ki,
            Vec::from_iter(0..problem.size()).as_slice(),
        );
        for (j, &kij) in status_ext.ki.iter().enumerate() {
            status_next.ka[j] -= kij * stepsize * dir.a[i] / problem.lambda();
            let rj = status.a[j] + status.g[j];
            if rj != 0.0 {
                pred_desc += kij * dir.a[i] * rj / problem.lambda();
            }
        }
    }
    let (obj1, _obj_dual) = problem.objective(&status_next);
    status_next.value = obj1;
    (pred_desc, status_next)
}

fn descent_step(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    params: &Params,
    status_ext: &mut StatusExtended,
    try_newton: bool,
) -> (DirectionType, f64, usize) {
    // compute Newton or gradient direction
    let mut direction_type = if try_newton {
        super::direction::newton_with_fallback(problem, kernel, status_ext)
    } else {
        super::direction::gradient(problem, kernel, status_ext);
        DirectionType::Gradient
    };
    let (obj0, _obj_dual) = problem.objective(&status_ext.status);
    let mut stepsize = 1.0;
    let mut backstep = 0;
    let desc = loop {
        let (pred_desc, status_next) = update_status(problem, kernel, status_ext, stepsize);
        if pred_desc < 0.0 {
            direction_type = DirectionType::NoStep;
            break 0.0;
        }
        let obj1 = status_next.value;
        let desc: f64 = obj0 - obj1;
        if desc > params.sigma * stepsize * pred_desc {
            status_ext.status = status_next;
            break desc;
        }
        stepsize *= params.eta;
        backstep += 1;
        if backstep >= params.max_back_steps {
            direction_type = DirectionType::NoStep;
            break 0.0;
        }
    };
    (direction_type, desc, backstep)
}

/// Uses a version of Newton's method to solve the given training problem starting from a particular [`Status`].
pub fn solve_with_status(
    status: Status,
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let start = Instant::now();
    let n = problem.size();
    let mut step: usize = 0;
    let mut stop = false;
    let mut final_step = false;
    let mut fresh_ka = false;
    let mut recompute_ka = false;

    let mut status_ext = StatusExtended {
        status: status.clone(),
        dir: Direction::new(n),
        active: ActiveSet::new(n),
        sums: Sums::new(),
        h: vec![0.0; n],
        ki: vec![0.0; n],
    };

    let (obj_primal, _obj_dual) = problem.objective(&status);
    status_ext.status.value = obj_primal;
    loop {
        // update steps and time
        status_ext.status.steps = step;
        let elapsed = start.elapsed().as_secs_f64();
        status_ext.status.time = elapsed;

        if final_step {
            status_ext.status.code = StatusCode::Optimal;
            stop = true;
        }

        // handle step limit
        if !stop && step >= params.max_steps {
            status_ext.status.code = StatusCode::MaxSteps;
            stop = true;
        }

        // handle time limit
        if !stop && params.time_limit > 0.0 && elapsed >= params.time_limit {
            status_ext.status.code = StatusCode::TimeLimit;
            stop = true;
        }

        // handle callback
        if let Some(callback_fn) = callback {
            if !stop && callback_fn(&status_ext.status) {
                status_ext.status.code = StatusCode::Callback;
                stop = true;
            }
        };

        // recompute ka if necessary
        if recompute_ka {
            recompute_ka = false;
            fresh_ka = true;
            problem.recompute_kernel_product(
                kernel,
                &mut status_ext.status,
                Vec::from_iter(0..n).as_slice(),
            );
        }

        // compute decisions
        compute_decisions(problem, &mut status_ext);

        // check for optimality
        let optimal = problem.is_optimal(&status_ext.status, params.tol);
        if optimal {
            if fresh_ka {
                final_step = true;
            } else {
                recompute_ka = true;
                continue;
            }
        }

        // update time
        let elapsed = start.elapsed().as_secs_f64();
        status_ext.status.time = elapsed;

        // handle progress output
        if params.verbose > 0 && stop {
            println!(
                "{:10} {:10.2} X {:3} {:10} {:10.03e} {:10.6} {:8.3}",
                step,
                elapsed,
                "",
                "",
                status_ext.status.violation,
                status_ext.status.value,
                status_ext.status.asum,
            )
        }

        // terminate
        if stop {
            break;
        }

        // find stepsize and update
        let (direction_type, desc, backstep) =
            descent_step(problem, kernel, params, &mut status_ext, true);

        // handle termination
        if matches!(direction_type, DirectionType::NoStep) {
            if !fresh_ka {
                recompute_ka = true;
            }
        } else {
            if desc.powi(2) < params.tol {
                if fresh_ka {
                    status_ext.status.code = StatusCode::Optimal;
                    stop = true;
                }
                recompute_ka = true;
            }
            fresh_ka = false;
        }

        if !final_step && matches!(direction_type, DirectionType::NoStep) && fresh_ka {
            let (_direction_type, desc, _backstep) =
                descent_step(problem, kernel, params, &mut status_ext, true);
            if desc.powi(2) < params.tol {
                status_ext.status.code = StatusCode::NoStepPossible;
                stop = true;
            }
        }

        // handle progress output
        if params.verbose > 0 && (step % params.verbose == 0 || optimal) {
            println!(
                "{:10} {:10.2} {} {:3} {:10} {:10.03e} {:10.03e} {:10.6} {:8.3}",
                step,
                elapsed,
                match direction_type {
                    DirectionType::Gradient => "G",
                    DirectionType::Newton => "N",
                    DirectionType::NoStep => "?",
                },
                backstep,
                status_ext.active.size_positive,
                status_ext.status.violation,
                desc,
                status_ext.status.value,
                status_ext.status.asum,
            )
        }
        step += 1;

        // terminate
        if final_step {
            status_ext.status.code = StatusCode::Optimal;
            stop = true;
        }
        if stop {
            break;
        }
    }
    status
}
