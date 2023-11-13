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

struct ActiveSet {
    size: usize,
    size_positive: usize,
    positive: Vec<usize>,
    zeros: Vec<usize>,
}

impl ActiveSet {
    fn new(size: usize) -> Self {
        Self {
            size,
            size_positive: 0,
            positive: Vec::new(),
            zeros: Vec::new(),
        }
    }

    fn size(&self) -> usize {
        self.size_positive
    }

    fn all(&self) -> &[usize] {
        &self.positive[..]
    }

    fn positives(&self) -> &[usize] {
        &self.positive[..self.size_positive]
    }

    fn zeros(&self) -> &[usize] {
        &self.positive[self.size_positive..]
    }

    fn make_full(&mut self) {
        self.positive = (0..self.size).collect();
    }

    fn merge(&mut self) {
        self.size_positive = self.positive.len();
        self.positive.append(&mut self.zeros);
    }
}

struct Direction {
    a: Vec<f64>,
    b: f64,
    c: f64,
}

impl Direction {
    fn new(size: usize) -> Self {
        Self {
            a: vec![0.0; size],
            b: 0.0,
            c: 0.0,
        }
    }
}

struct Sums {
    a: f64,
    g: f64,
    da_zeros: f64,
}

impl Sums {
    fn new() -> Self {
        Self {
            a: 0.0,
            g: 0.0,
            da_zeros: 0.0,
        }
    }
}

fn compute_decisions(
    problem: &dyn Problem,
    status: &mut Status,
    h: &mut Vec<f64>,
    dir: &mut Direction,
) -> (Sums, ActiveSet) {
    let mut active = ActiveSet::new(problem.size());
    let mut sums = Sums::new();
    let mut violation = 0.0;
    let mut abs_asum = 0.0;
    for i in 0..problem.size() {
        let ai = status.a[i];
        sums.a += ai;
        abs_asum += problem.sign(i) * ai;
        let ti = status.ka[i] + status.b + status.c * problem.sign(i);
        let gi = problem.d_loss(i, ti);
        status.g[i] = gi;
        sums.g += gi;
        violation += (ai + gi).abs();
        let hi = problem.d2_loss(i, ti);
        h[i] = hi;
        if h[i] == 0.0 {
            let dai = ai + gi;
            dir.a[i] = dai;
            if dai != 0.0 {
                active.zeros.push(i);
            }
            sums.da_zeros += dai;
        } else {
            active.positive.push(i);
        }
    }
    status.violation = violation + sums.a.abs();
    status.asum = abs_asum;
    (sums, active)
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
    let mut fresh_ka = false;
    let mut recompute_ka = false;
    let mut last_step_descent = false;

    let (obj_primal, _obj_dual) = problem.objective(&status);
    status.value = obj_primal;
    let mut h = vec![0.0; n];
    let mut dir = Direction::new(n);
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

        // recompute ka if necessary
        if recompute_ka {
            recompute_ka = false;
            fresh_ka = true;
            let full_set = (0..n).collect();
            problem.recompute_kernel_product(kernel, &mut status, &full_set);
        }

        // compute decisions
        let (sums, mut active) = compute_decisions(problem, &mut status, &mut h, &mut dir);

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

        let n_active = active.positive.len();

        // TODO: think about the size of ki
        let mut ki = vec![0.0; n];
        let gradient_step = n_active == 0;
        if n_active == 0 {
            // compute gradient direction
            for i in 0..n {
                dir.a[i] = status.a[i] + status.g[i];
            }
            dir.b = sums.g / problem.lambda();
            active.make_full();
        } else {
            // compute Newton direction
            active.merge();
            let mut mat = Array::zeros((n_active, n_active));
            let mut rhs = Array::zeros((n_active,));
            let mut rhs_i;
            for (idx_i, &i) in active.positives().iter().enumerate() {
                kernel.compute_row(i, &mut ki, active.all());
                for idx_j in 0..n_active {
                    mat[(idx_i, idx_j)] = ki[idx_j] / problem.lambda();
                    if idx_i == idx_j {
                        mat[(idx_i, idx_j)] += 1.0 / h[i];
                    }
                }
                rhs_i = (status.a[i] + status.g[i]) / h[i];
                for (idx_j, &j) in active.zeros().iter().enumerate() {
                    rhs_i -= dir.a[j] * ki[n_active + idx_j] / problem.lambda();
                }
                rhs[idx_i] = rhs_i;
            }
            let mat_fact = mat.factorize_into().unwrap();
            let mat_inv_rhs = mat_fact.solve_into(rhs).unwrap();
            let mat_inv_one = mat_fact.solve_into(Array::ones((n_active,))).unwrap();

            let rhs_b = sums.a - sums.da_zeros;
            dir.b = (mat_inv_rhs.sum() - rhs_b) / mat_inv_one.sum();
            let da_nonzero = mat_inv_rhs - dir.b * mat_inv_one;
            for (idx_i, &i) in active.positives().iter().enumerate() {
                dir.a[i] = da_nonzero[idx_i];
            }
        }

        let mut stepsize = 1.0;

        let (obj0, _obj_dual) = problem.objective(&status);
        let mut status_next = status.clone();
        let mut backstep = 0;
        for _backstep in 0..params.max_back_steps {
            let mut pred_desc = sums.g * dir.b;
            status_next.b -= stepsize * dir.b;
            // TODO: think about the use of vector da
            for &i in active.all().iter() {
                if dir.a[i] == 0.0 {
                    continue;
                }
                status_next.a[i] -= stepsize * dir.a[i];
                kernel.compute_row(i, &mut ki, Vec::from_iter(0..n).as_slice());
                for (j, &kij) in ki.iter().enumerate() {
                    status_next.ka[j] -= kij * stepsize * dir.a[i] / problem.lambda();
                    let rj = status.a[j] + status.g[j];
                    if rj != 0.0 {
                        pred_desc += kij * dir.a[i] * rj / problem.lambda();
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
                active.size_positive,
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
