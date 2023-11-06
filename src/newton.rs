use ndarray::prelude::*;
use ndarray_linalg::{FactorizeInto, Solve};

use crate::kernel::Kernel;
use crate::problem::Problem;
use std::time::Instant;

use super::status::{Status, StatusCode};

#[derive(Debug)]
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
    let n = problem.size();
    let status = Status::new(n);
    solve_with_status(status, problem, kernel, params, callback)
}
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

    // status.dec = vec![0.0; n];
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
        let mut active_set = Vec::new();
        let mut active_zeros = Vec::new();
        let mut dasum_zeros = 0.0;
        let mut violation = 0.0;
        let mut asum = 0.0;
        for i in 0..problem.size() {
            let ai = status.a[i];
            asum += ai;
            let ti = status.ka[i] + status.b + status.c * problem.sign(i);
            let gi = problem.d_loss(i, ti);
            status.g[i] = gi;
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
        status.asum = asum;

        let n_active = active_set.len();
        let mut mat = Array::zeros((n_active, n_active));
        let mut rhs = Array::zeros((n_active,));

        // compute Newton direction
        let mut rhs_i;
        // TODO: think about the size of ki
        let mut ki = vec![0.0; n];
        active_set.append(&mut active_zeros);
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
            rhs[idx_i] = rhs_i / h[i];
        }
        let mat_fact = mat.factorize_into().unwrap();
        let mat_inv_rhs = mat_fact.solve_into(rhs).unwrap();
        let mat_inv_one = mat_fact.solve_into(Array::ones((n_active,))).unwrap();

        let rhs_b = asum - dasum_zeros;
        let db = (mat_inv_rhs.sum() - rhs_b) / mat_inv_one.sum();
        let da_nonzero = mat_inv_rhs - db * mat_inv_one;
        for (idx_i, &i) in active_set[..n_active].iter().enumerate() {
            da[i] = da_nonzero[idx_i];
        }

        let mut stepsize = 1.0;

        let (obj0, _obj_dual) = problem.objective(&status);
        println!("objective {obj0}");
        let mut status_next = status.clone();
        for _backstep in 0..10 {
            status_next.b -= stepsize * db;
            // TODO: think about the use of vector da
            for &i in active_set.iter() {
                status_next.a[i] -= stepsize * da[i];
                kernel.compute_row(i, &mut ki, &(0..n).collect());
                for (j, &kij) in ki.iter().enumerate() {
                    status_next.ka[j] -= kij * stepsize * da[i];
                }
            }
            let (obj1, _obj_dual) = problem.objective(&status);
            if obj1 < obj0 {
                break;
            }
            stepsize *= 0.5;
            status_next = status.clone();
        }
        status = status_next;
        step += 1;
    }
    status
}
