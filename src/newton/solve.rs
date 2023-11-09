use ndarray::prelude::*;
use ndarray_linalg::{FactorizeInto, Solve};
use std::time::Instant;

use super::params::Params;
use crate::kernel::Kernel;
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

/// Uses a version of Newton's method to solve the given training problem starting from a particular [`Status`].
pub fn solve_with_status(
    status: Status,
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    params: &Params,
    callback: Option<&dyn Fn(&Status) -> bool>,
) -> Status {
    let mut status = status;
    let start = Instant::now();
    let n = problem.size();
    let mut step: usize = 0;
    let mut stop = false;
    let mut last_step_descent = false;

    let (obj_primal, _obj_dual) = problem.objective(&status);
    status.value = obj_primal;
    let mut h = vec![0.0; n];
    let mut da = vec![0.0; n];
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

        // compute decisions
        let mut active_set = Vec::new();
        let mut active_zeros = Vec::new();
        let mut dasum_zeros = 0.0;
        let mut violation = 0.0;
        let mut asum = 0.0;
        let mut abs_asum = 0.0;
        let mut gsum = 0.0;
        for i in 0..problem.size() {
            let ai = status.a[i];
            asum += ai;
            abs_asum += problem.sign(i) * ai;
            let ti = status.ka[i] + status.b + status.c * problem.sign(i);
            let gi = problem.d_loss(i, ti);
            status.g[i] = gi;
            gsum += gi;
            violation += (ai + gi).abs();
            let hi = problem.d2_loss(i, ti);
            h[i] = hi;
            if h[i] == 0.0 {
                let dai = ai + gi;
                da[i] = dai;
                if dai != 0.0 {
                    active_zeros.push(i);
                }
                dasum_zeros += dai;
            } else {
                active_set.push(i);
            }
        }
        status.violation = violation + asum.abs();
        status.asum = abs_asum;

        // check for optimality
        let optimal = problem.is_optimal(&status, params.tol);
        if optimal {
            status.code = StatusCode::Optimal;
            stop = true;
        }

        // update time
        let elapsed = start.elapsed().as_secs_f64();
        status.time = elapsed;

        // handle progress output
        if params.verbose > 0 && stop {
            println!(
                "{:10} {:10.2} X {:3} {:10} {:10.03e} {:10.6} {:8.3}",
                step, elapsed, "", "", status.violation, status.value, status.asum,
            )
        }

        // terminate
        if stop {
            break;
        }

        let n_active = active_set.len();

        // TODO: think about the size of ki
        let mut ki = vec![0.0; n];
        let db;
        let gradient_step = n_active == 0;
        if n_active == 0 {
            for i in 0..n {
                da[i] = status.a[i] + status.g[i];
            }
            db = gsum / problem.lambda();
            active_set = (0..n).collect();
        } else {
            // compute Newton direction
            active_set.append(&mut active_zeros);
            let mut mat = Array::zeros((n_active, n_active));
            let mut rhs = Array::zeros((n_active,));
            let mut rhs_i;
            for (idx_i, &i) in active_set[..n_active].iter().enumerate() {
                kernel.compute_row(i, &mut ki, &active_set);
                for idx_j in 0..n_active {
                    mat[(idx_i, idx_j)] = ki[idx_j] / problem.lambda();
                    if idx_i == idx_j {
                        mat[(idx_i, idx_j)] += 1.0 / h[i];
                    }
                }
                rhs_i = (status.a[i] + status.g[i]) / h[i];
                for (idx_j, &j) in active_set[n_active..].iter().enumerate() {
                    rhs_i -= da[j] * ki[n_active + idx_j] / problem.lambda();
                }
                rhs[idx_i] = rhs_i;
            }
            let mat_fact = mat.factorize_into().unwrap();
            let mat_inv_rhs = mat_fact.solve_into(rhs).unwrap();
            let mat_inv_one = mat_fact.solve_into(Array::ones((n_active,))).unwrap();

            let rhs_b = asum - dasum_zeros;
            db = (mat_inv_rhs.sum() - rhs_b) / mat_inv_one.sum();
            let da_nonzero = mat_inv_rhs - db * mat_inv_one;
            for (idx_i, &i) in active_set[..n_active].iter().enumerate() {
                da[i] = da_nonzero[idx_i];
            }
        }

        let mut stepsize = 1.0;

        let (obj0, _obj_dual) = problem.objective(&status);
        let mut status_next = status.clone();
        let mut backstep = 0;
        for _backstep in 0..params.max_back_steps {
            let mut pred_desc = gsum * db;
            status_next.b -= stepsize * db;
            // TODO: think about the use of vector da
            for &i in active_set.iter() {
                if da[i] == 0.0 {
                    continue;
                }
                status_next.a[i] -= stepsize * da[i];
                kernel.compute_row(i, &mut ki, &(0..n).collect());
                for (j, &kij) in ki.iter().enumerate() {
                    status_next.ka[j] -= kij * stepsize * da[i] / problem.lambda();
                    let rj = status.a[j] + status.g[j];
                    if rj != 0.0 {
                        pred_desc += kij * da[i] * rj / problem.lambda();
                    }
                }
            }
            let (obj1, _obj_dual) = problem.objective(&status_next);
            status_next.value = obj1;
            let desc: f64 = obj0 - obj1;
            if desc > params.sigma * pred_desc {
                break;
            }
            stepsize *= params.eta;
            status_next = status.clone();
            backstep += 1;
        }
        // handle progress output
        if params.verbose > 0 && (step % params.verbose == 0 || optimal) {
            println!(
                "{:10} {:10.2} {} {:3} {:10} {:10.03e} {:10.6} {:8.3}",
                step,
                elapsed,
                if gradient_step { "G" } else { "N" },
                backstep,
                active_set.len(),
                status.violation,
                status.value,
                status.asum,
            )
        }
        if backstep == params.max_back_steps {
            if last_step_descent {
                status.code = StatusCode::Optimal;
                stop = true;
            } else {
                last_step_descent = true;
                let full_set = (0..n).collect();
                problem.recompute_kernel_product(kernel, &mut status, &full_set);
            }
        } else {
            last_step_descent = false;
            status = status_next;
        }

        // terminate
        if stop {
            break;
        }
        step += 1;
    }
    status
}
