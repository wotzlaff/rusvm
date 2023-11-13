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

struct StatusExtended {
    status: Status,
    dir: Direction,
    active: ActiveSet,
    sums: Sums,
    h: Vec<f64>,
    ki: Vec<f64>,
}

fn compute_decisions(problem: &dyn Problem, status_ext: &mut StatusExtended) {
    let mut active = ActiveSet::new(problem.size());
    let mut sums = Sums::new();
    let mut violation = 0.0;
    let mut abs_asum = 0.0;
    for i in 0..problem.size() {
        let ai = status_ext.status.a[i];
        sums.a += ai;
        abs_asum += problem.sign(i) * ai;
        let ti =
            status_ext.status.ka[i] + status_ext.status.b + status_ext.status.c * problem.sign(i);
        let gi = problem.d_loss(i, ti);
        status_ext.status.g[i] = gi;
        sums.g += gi;
        violation += (ai + gi).abs();
        let hi = problem.d2_loss(i, ti);
        status_ext.h[i] = hi;
        if hi == 0.0 {
            let dai = ai + gi;
            status_ext.dir.a[i] = dai;
            if dai != 0.0 {
                active.zeros.push(i);
            }
            sums.da_zeros += dai;
        } else {
            active.positive.push(i);
        }
    }
    status_ext.status.violation = violation + sums.a.abs();
    status_ext.status.asum = abs_asum;
    status_ext.active = active;
    status_ext.sums = sums;
}

fn compute_newton_direction(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    status_ext: &mut StatusExtended,
) {
    let h: &Vec<f64> = &status_ext.h;
    let sums = &status_ext.sums;
    let active = &status_ext.active;
    let n_active = active.size_positive;
    let mut mat = Array::zeros((n_active, n_active));
    let mut rhs = Array::zeros((n_active,));
    let mut rhs_i;
    for (idx_i, &i) in active.positives().iter().enumerate() {
        kernel.compute_row(i, &mut status_ext.ki, active.all());
        for idx_j in 0..n_active {
            mat[(idx_i, idx_j)] = status_ext.ki[idx_j] / problem.lambda();
            if idx_i == idx_j {
                mat[(idx_i, idx_j)] += 1.0 / h[i];
            }
        }
        rhs_i = (status_ext.status.a[i] + status_ext.status.g[i]) / h[i];
        for (idx_j, &j) in active.zeros().iter().enumerate() {
            rhs_i -= status_ext.dir.a[j] * status_ext.ki[n_active + idx_j] / problem.lambda();
        }
        rhs[idx_i] = rhs_i;
    }
    let mat_fact = mat.factorize_into().unwrap();
    let mat_inv_rhs = mat_fact.solve_into(rhs).unwrap();
    let mat_inv_one = mat_fact.solve_into(Array::ones((n_active,))).unwrap();

    let rhs_b = sums.a - sums.da_zeros;
    let db = (mat_inv_rhs.sum() - rhs_b) / mat_inv_one.sum();
    status_ext.dir.b = db;
    let da_nonzero = mat_inv_rhs - db * mat_inv_one;
    for (idx_i, &i) in active.positives().iter().enumerate() {
        status_ext.dir.a[i] = da_nonzero[idx_i];
    }
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
    status_next.b -= stepsize * dir.b;
    // TODO: think about the use of vector da
    for &i in status_ext.active.all().iter() {
        if dir.a[i] == 0.0 {
            continue;
        }
        status_next.a[i] -= stepsize * dir.a[i];
        kernel.compute_row(
            i,
            &mut status_ext.ki,
            Vec::from_iter(0..status_ext.active.size).as_slice(),
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
    let mut fresh_ka = false;
    let mut recompute_ka = false;
    let mut last_step_descent = false;

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

        // handle step limit
        if step >= params.max_steps {
            status_ext.status.code = StatusCode::MaxSteps;
            stop = true;
        }

        // handle time limit
        if params.time_limit > 0.0 && elapsed >= params.time_limit {
            status_ext.status.code = StatusCode::TimeLimit;
            stop = true;
        }

        // handle callback
        if let Some(callback_fn) = callback {
            if callback_fn(&status_ext.status) {
                status_ext.status.code = StatusCode::Callback;
                stop = true;
            }
        };

        // recompute ka if necessary
        if recompute_ka {
            recompute_ka = false;
            fresh_ka = true;
            let full_set = (0..n).collect();
            problem.recompute_kernel_product(kernel, &mut status_ext.status, &full_set);
        }

        // compute decisions
        compute_decisions(problem, &mut status_ext);

        // check for optimality
        let optimal = problem.is_optimal(&status_ext.status, params.tol);
        if optimal {
            status_ext.status.code = StatusCode::Optimal;
            stop = true;
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

        // TODO: think about the size of ki
        let n_active = status_ext.active.positive.len();
        let gradient_step = n_active == 0;
        if n_active == 0 {
            // compute gradient direction
            status_ext.active.make_full();
            for i in 0..n {
                status_ext.dir.a[i] = status_ext.status.a[i] + status_ext.status.g[i];
            }
            status_ext.dir.b = status_ext.sums.g / problem.lambda();
        } else {
            // compute Newton direction
            status_ext.active.merge();
            compute_newton_direction(problem, kernel, &mut status_ext);
        }

        let mut stepsize = 1.0;

        let (obj0, _obj_dual) = problem.objective(&status_ext.status);
        let mut backstep = 0;
        let status_next = loop {
            let (pred_desc, status_next) =
                update_status(problem, kernel, &mut status_ext, stepsize);
            let obj1 = status_next.value;
            let desc: f64 = obj0 - obj1;
            if desc > params.sigma * pred_desc {
                break status_next;
            }
            stepsize *= params.eta;
            backstep += 1;
            if backstep >= params.max_back_steps {
                break status_ext.status.clone();
            }
        };
        // handle progress output
        if params.verbose > 0 && (step % params.verbose == 0 || optimal) {
            println!(
                "{:10} {:10.2} {} {:3} {:10} {:10.03e} {:10.6} {:8.3}",
                step,
                elapsed,
                if gradient_step { "G" } else { "N" },
                backstep,
                status_ext.active.size_positive,
                status_ext.status.violation,
                status_ext.status.value,
                status_ext.status.asum,
            )
        }
        if backstep == params.max_back_steps {
            if last_step_descent {
                status_ext.status.code = StatusCode::Optimal;
                stop = true;
            } else {
                last_step_descent = true;
                let full_set = (0..n).collect();
                problem.recompute_kernel_product(kernel, &mut status_ext.status, &full_set);
            }
        } else {
            last_step_descent = false;
            status_ext.status = status_next;
        }

        // terminate
        if stop {
            break;
        }
        step += 1;
    }
    status
}
