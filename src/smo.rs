use crate::kernel::Kernel;
use crate::problem::Problem;
use crate::status::{Status, StatusCode};
use std::time::Instant;

fn find_mvp_signed(
    problem: &dyn Problem,
    status: &mut Status,
    active_set: &Vec<usize>,
    sign: f64,
) -> (f64, f64, usize, usize) {
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut idx_i: usize = 0;
    let mut idx_j: usize = 0;
    for (idx, &i) in active_set.iter().enumerate() {
        let g_i = problem.grad(status, i);
        status.g[i] = g_i;
        if problem.sign(i) * sign >= 0.0 {
            if status.a[i] > problem.lb(i) && g_i > g_max {
                idx_i = idx;
                g_max = g_i;
            }
            if status.a[i] < problem.ub(i) && g_i < g_min {
                idx_j = idx;
                g_min = g_i;
            }
        }
    }
    (g_max - g_min, g_max + g_min, idx_i, idx_j)
}

fn find_mvp(problem: &dyn Problem, status: &mut Status, active_set: &Vec<usize>) -> (usize, usize) {
    let (dij, sij, idx_i, idx_j) = find_mvp_signed(problem, status, active_set, 0.0);
    status.b = -0.5 * sij;
    status.violation = dij;
    (idx_i, idx_j)
}

fn descent(q: f64, p: f64, t_max: f64, lmbda: f64, regularization: f64) -> f64 {
    let t = f64::min(lmbda * p / f64::max(q, regularization), t_max);
    t * (p - 0.5 / lmbda * q * t)
}

fn find_ws2(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    idx_i0: usize,
    idx_j1: usize,
    status: &Status,
    active_set: &Vec<usize>,
    sign: f64,
) -> (usize, usize) {
    let i0 = active_set[idx_i0];
    let j1 = active_set[idx_j1];
    let gi0 = status.g[i0];
    let gj1 = status.g[j1];
    let mut max_d0 = 0.0;
    let mut max_d1 = 0.0;
    let mut idx_j0 = idx_j1;
    let mut idx_i1 = idx_i0;

    let diags: Vec<f64> = active_set.iter().map(|&i| kernel.diag(i)).collect();
    kernel.use_rows([i0, j1].to_vec(), &active_set, &mut |kij: Vec<&[f64]>| {
        let ki0 = kij[0];
        let kj1 = kij[1];
        let ki0i0 = ki0[idx_i0];
        let kj1j1 = kj1[idx_j1];
        let max_ti0 = status.a[i0] - problem.lb(i0);
        let max_tj1 = problem.ub(j1) - status.a[j1];

        for (idx_r, &r) in active_set.iter().enumerate() {
            if sign * problem.sign(r) < 0.0 {
                continue;
            }
            let gr = status.g[r];
            let krr = diags[idx_r];

            let pi0r = gi0 - gr;
            let d_upr = problem.ub(r) - status.a[r];
            if d_upr > 0.0 && pi0r > 0.0 {
                let qi0 = ki0i0 + krr - 2.0 * ki0[idx_r]
                    + problem.quad(status, i0)
                    + problem.quad(status, r);
                let di0r = descent(
                    qi0,
                    pi0r,
                    f64::min(max_ti0, d_upr),
                    problem.lambda(),
                    problem.regularization(),
                );
                if di0r > max_d0 {
                    idx_j0 = idx_r;
                    max_d0 = di0r;
                }
            }

            let prj1 = gr - gj1;
            let d_dnr = status.a[r] - problem.lb(r);
            if d_dnr > 0.0 && prj1 > 0.0 {
                let qj1 = kj1j1 + krr - 2.0 * kj1[idx_r]
                    + problem.quad(status, j1)
                    + problem.quad(status, r);
                let drj1 = descent(
                    qj1,
                    prj1,
                    f64::min(max_tj1, d_dnr),
                    problem.lambda(),
                    problem.regularization(),
                );
                if drj1 > max_d1 {
                    idx_i1 = idx_r;
                    max_d1 = drj1;
                }
            }
        }
    });
    if max_d0 > max_d1 {
        (idx_i0, idx_j0)
    } else {
        (idx_i1, idx_j1)
    }
}

fn update(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    idx_i: usize,
    idx_j: usize,
    status: &mut Status,
    active_set: &Vec<usize>,
) {
    let i = active_set[idx_i];
    let j = active_set[idx_j];
    kernel.use_rows([i, j].to_vec(), &active_set, &mut |kij: Vec<&[f64]>| {
        let ki = kij[0];
        let kj = kij[1];
        let pij = status.g[i] - status.g[j];
        let qij = ki[idx_i] + kj[idx_j] - 2.0 * ki[idx_j]
            + problem.quad(status, i)
            + problem.quad(status, j);
        let max_tij = f64::min(status.a[i] - problem.lb(i), problem.ub(j) - status.a[j]);
        let tij: f64 = f64::min(
            problem.lambda() * pij / f64::max(qij, problem.regularization()),
            max_tij,
        );
        status.a[i] -= tij;
        status.a[j] += tij;
        let tij_l = tij / problem.lambda();
        status.value -= tij * (0.5 * qij * tij_l - pij);
        for (idx, &k) in active_set.iter().enumerate() {
            status.ka[k] += tij_l * (kj[idx] - ki[idx]);
        }
    });
}

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
) -> Status {
    let start = Instant::now();
    let n = problem.size();
    let mut active_set = (0..n).collect();

    let mut status = Status::new(n);
    let mut step: usize = 0;
    let mut stop = false;
    loop {
        if step >= max_steps {
            status.code = StatusCode::MaxSteps;
            stop = true;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if time_limit > 0.0 && elapsed >= time_limit {
            status.code = StatusCode::TimeLimit;
            stop = true;
        }
        // TODO: callback
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
                    "{:10} {:10.2} {:10.6} {:10.6} {:10.6} {:10.6} {:8} / {}",
                    step,
                    elapsed,
                    status.violation,
                    gap,
                    obj_primal,
                    -obj_dual,
                    active_set.len(),
                    problem.size()
                )
            } else {
                println!(
                    "{:10} {:10.2} {:10.6} {:10.6} {:8} / {}",
                    step,
                    elapsed,
                    status.violation,
                    status.value,
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
            find_ws2(problem, kernel, idx_i0, idx_j1, &status, &active_set, 0.0)
        } else {
            (idx_i0, idx_j1)
        };
        update(problem, kernel, idx_i, idx_j, &mut status, &active_set);
        step += 1;
    }
    status.steps = step;
    status.time = start.elapsed().as_secs_f64();
    status
}
