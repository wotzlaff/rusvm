use crate::kernel::Kernel;
use crate::problem::Problem;
use crate::status::{Status, StatusCode};
use std::time::Instant;

fn find_mvp_signed(
    problem: &impl Problem,
    state: &mut Status,
    active_set: &Vec<usize>,
    sign: f64,
) -> (f64, f64, usize, usize) {
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut idx_i: usize = 0;
    let mut idx_j: usize = 0;
    for (idx, &i) in active_set.iter().enumerate() {
        let g_i = problem.grad(state, i);
        state.g[i] = g_i;
        if problem.sign(i) * sign >= 0.0 {
            if state.a[i] > problem.lb(i) && g_i > g_max {
                idx_i = idx;
                g_max = g_i;
            }
            if state.a[i] < problem.ub(i) && g_i < g_min {
                idx_j = idx;
                g_min = g_i;
            }
        }
    }
    (g_max - g_min, g_max + g_min, idx_i, idx_j)
}

fn find_mvp(problem: &impl Problem, state: &mut Status, active_set: &Vec<usize>) -> (usize, usize) {
    let (dij, sij, idx_i, idx_j) = find_mvp_signed(problem, state, active_set, 0.0);
    state.b = -0.5 * sij;
    state.violation = dij;
    (idx_i, idx_j)
}

fn descent(q: f64, p: f64, t_max: f64, lmbda: f64, regularization: f64) -> f64 {
    let t = f64::min(lmbda * p / f64::max(q, regularization), t_max);
    t * (p - 0.5 / lmbda * q * t)
}

fn find_ws2(
    problem: &impl Problem,
    kernel: &mut dyn Kernel,
    idx_i0: usize,
    idx_j1: usize,
    state: &Status,
    active_set: &Vec<usize>,
    sign: f64,
) -> (usize, usize) {
    let i0 = active_set[idx_i0];
    let j1 = active_set[idx_j1];
    let gi0 = state.g[i0];
    let gj1 = state.g[j1];
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
        let max_ti0 = state.a[i0] - problem.lb(i0);
        let max_tj1 = problem.ub(j1) - state.a[j1];

        for (idx_r, &r) in active_set.iter().enumerate() {
            if sign * problem.sign(r) < 0.0 {
                continue;
            }
            let gr = state.g[r];
            let krr = diags[idx_r];

            let pi0r = gi0 - gr;
            let d_upr = problem.ub(r) - state.a[r];
            if d_upr > 0.0 && pi0r > 0.0 {
                let qi0 = ki0i0 + krr - 2.0 * ki0[idx_r]
                    + problem.quad(state, i0)
                    + problem.quad(state, r);
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
            let d_dnr = state.a[r] - problem.lb(r);
            if d_dnr > 0.0 && prj1 > 0.0 {
                let qj1 = kj1j1 + krr - 2.0 * kj1[idx_r]
                    + problem.quad(state, j1)
                    + problem.quad(state, r);
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
    problem: &impl Problem,
    kernel: &mut dyn Kernel,
    idx_i: usize,
    idx_j: usize,
    state: &mut Status,
    active_set: &Vec<usize>,
) {
    let i = active_set[idx_i];
    let j = active_set[idx_j];
    kernel.use_rows([i, j].to_vec(), &active_set, &mut |kij: Vec<&[f64]>| {
        let ki = kij[0];
        let kj = kij[1];
        let pij = state.g[i] - state.g[j];
        let qij = ki[idx_i] + kj[idx_j] - 2.0 * ki[idx_j]
            + problem.quad(state, i)
            + problem.quad(state, j);
        let max_tij = f64::min(state.a[i] - problem.lb(i), problem.ub(j) - state.a[j]);
        let tij: f64 = f64::min(
            problem.lambda() * pij / f64::max(qij, problem.regularization()),
            max_tij,
        );
        state.a[i] -= tij;
        state.a[j] += tij;
        let tij_l = tij / problem.lambda();
        state.value -= tij * (0.5 * qij * tij_l - pij);
        for (idx, &k) in active_set.iter().enumerate() {
            state.ka[k] += tij_l * (kj[idx] - ki[idx]);
        }
    });
}

pub fn solve(
    problem: &impl Problem,
    kernel: &mut dyn Kernel,
    tol: f64,
    max_steps: usize,
    verbose: usize,
    second_order: bool,
    shrinking_period: usize,
    shrinking_threshold: f64,
    time_limit: f64,
) -> Status {
    let start = Instant::now();
    let n = problem.size();
    let mut state = Status::new(n);
    let mut active_set = (0..n).collect();

    let mut status = Status::new(n);
    let mut step: usize = 0;
    let mut stop = false;
    loop {
        if step >= max_steps {
            stop = true;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if time_limit > 0.0 && elapsed >= time_limit {
            status.code = StatusCode::TimeLimit;
            stop = true;
        }
        // TODO: callback
        if shrinking_period > 0 && step % shrinking_period == 0 {
            problem.shrink(kernel, &state, &mut active_set, shrinking_threshold);
        }

        let (idx_i0, idx_j1) = find_mvp(problem, &mut state, &active_set);
        let optimal = problem.is_optimal(&state, tol);

        if verbose > 0 && (step % verbose == 0 || optimal) {
            println!(
                "{:10} {:10.2} {:10.6} {:10.6} {} / {}",
                step,
                elapsed,
                state.violation,
                state.value,
                active_set.len(),
                problem.size()
            )
        }

        if optimal {
            if problem.is_shrunk(&state, &active_set) {
                problem.unshrink(kernel, &mut state, &mut active_set);
                continue;
            }
            status.code = StatusCode::Optimal;
            stop = true;
        }

        if stop {
            break;
        }

        let (idx_i, idx_j) = if second_order {
            find_ws2(problem, kernel, idx_i0, idx_j1, &state, &active_set, 0.0)
        } else {
            (idx_i0, idx_j1)
        };
        update(problem, kernel, idx_i, idx_j, &mut state, &active_set);
        step += 1;
    }
    status.steps = step;
    status.time = start.elapsed().as_secs_f64();
    status
}
