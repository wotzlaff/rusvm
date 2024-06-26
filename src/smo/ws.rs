use super::subproblem::{compute_step, Subproblem};
use crate::kernel::Kernel;
use crate::problem::DualProblem;
use crate::status::Status;

fn find_mvp_signed(
    problem: &dyn DualProblem,
    status: &mut Status,
    active_set: &Vec<usize>,
    sign: f64,
) -> (f64, f64, usize, usize) {
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut idx_i: usize = 0;
    let mut idx_j: usize = 0;
    for (idx, &i) in active_set.iter().enumerate() {
        let g_i = status.ka[i] + problem.d_dloss(i, status.a[i]);
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

pub fn find_mvp(
    problem: &dyn DualProblem,
    status: &mut Status,
    active_set: &Vec<usize>,
) -> (usize, usize) {
    let (dij, idx_i, idx_j) = if status.asum == problem.max_asum() {
        let (dij_p, sij_p, idx_i_p, idx_j_p) = find_mvp_signed(problem, status, active_set, 1.0);
        let (dij_n, sij_n, idx_i_n, idx_j_n) = find_mvp_signed(problem, status, active_set, -1.0);
        status.b = -0.25 * (sij_p + sij_n);
        status.c = 0.25 * (sij_n - sij_p);
        if dij_p >= dij_n {
            (dij_p, idx_i_p, idx_j_p)
        } else {
            (dij_n, idx_i_n, idx_j_n)
        }
    } else {
        let (dij, sij, idx_i, idx_j) = find_mvp_signed(problem, status, active_set, 0.0);
        status.b = -0.5 * sij;
        status.opt_status.violation = dij;
        (dij, idx_i, idx_j)
    };
    status.opt_status.violation = dij;
    (idx_i, idx_j)
}

pub fn find_ws2(
    problem: &dyn DualProblem,
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
    kernel.use_rows([i0, j1].as_slice(), &active_set, &mut |kij: Vec<&[f64]>| {
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
                let step = compute_step(
                    problem,
                    Subproblem {
                        ij: (i0, r),
                        max_t: f64::min(max_ti0, d_upr),
                        q0: ki0i0 + krr - 2.0 * ki0[idx_r],
                        p0: status.ka[i0] - status.ka[r],
                    },
                    status,
                );
                if step.dvalue > max_d0 {
                    idx_j0 = idx_r;
                    max_d0 = step.dvalue;
                }
            }

            let prj1 = gr - gj1;
            let d_dnr = status.a[r] - problem.lb(r);
            if d_dnr > 0.0 && prj1 > 0.0 {
                let step = compute_step(
                    problem,
                    Subproblem {
                        ij: (r, j1),
                        max_t: f64::min(max_tj1, d_dnr),
                        q0: kj1j1 + krr - 2.0 * kj1[idx_r],
                        p0: status.ka[r] - status.ka[j1],
                    },
                    status,
                );
                if step.dvalue > max_d1 {
                    idx_i1 = idx_r;
                    max_d1 = step.dvalue;
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
