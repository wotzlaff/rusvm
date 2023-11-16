use super::newton::newton;
use crate::kernel::Kernel;
use crate::problem::DualProblem;
use crate::status::Status;

struct Step {
    t: f64,
    dvalue: f64,
}

struct SubProblem {
    ij: (usize, usize),
    max_t: f64,
    q0: f64,
    p0: f64,
}

fn compute_step(problem: &dyn DualProblem, sprob: SubProblem, status: &Status) -> Step {
    let (i, j) = sprob.ij;
    let ai = status.a[i];
    let aj = status.a[j];
    if problem.is_quad() {
        let p = sprob.p0 + problem.d_dloss(i, ai) - problem.d_dloss(j, aj);
        let q = sprob.q0 + problem.d2_dloss(i, ai) + problem.d2_dloss(j, aj);
        let t = f64::min(p / f64::max(q, problem.regularization()), sprob.max_t);
        let dvalue = t * (0.5 * q * t - p);
        Step { t, dvalue }
    } else {
        let loss = problem.dloss(i, ai) + problem.dloss(j, aj);
        let (t, dvalue) = newton(
            &|t| {
                let v = t * (0.5 * sprob.q0 * t - sprob.p0) - loss
                    + problem.dloss(i, ai - t)
                    + problem.dloss(j, aj + t);
                let dv = sprob.q0 * t - sprob.p0 - problem.d_dloss(i, ai - t)
                    + problem.d_dloss(j, aj + t);
                let ddv: f64 = sprob.q0 + problem.d2_dloss(i, ai - t) + problem.d2_dloss(j, aj + t);
                (v, f64::max(dv, -1e3), f64::min(ddv, 1e6))
            },
            0.0,
            sprob.max_t,
        );
        Step { t, dvalue }
    }
}

pub fn update(
    problem: &dyn DualProblem,
    kernel: &mut dyn Kernel,
    idx_i: usize,
    idx_j: usize,
    status: &mut Status,
    active_set: &Vec<usize>,
) {
    let i = active_set[idx_i];
    let j = active_set[idx_j];
    kernel.use_rows([i, j].as_slice(), &active_set, &mut |kij: Vec<&[f64]>| {
        let ki = kij[0];
        let kj = kij[1];
        let mut max_tij = f64::min(status.a[i] - problem.lb(i), problem.ub(j) - status.a[j]);

        let max_t_asum = 0.5 * (problem.max_asum() - status.asum);
        let update_asum = if problem.has_max_asum() {
            if problem.sign(i) != problem.sign(j) {
                if max_tij > max_t_asum {
                    max_tij = max_t_asum;
                }
                true
            } else {
                false
            }
        } else {
            false
        };
        let step = compute_step(
            problem,
            SubProblem {
                ij: (i, j),
                max_t: max_tij,
                q0: (ki[idx_i] + kj[idx_j] - 2.0 * ki[idx_j]) / problem.lambda(),
                p0: status.ka[i] - status.ka[j],
            },
            &status,
        );

        let t = step.t;
        if update_asum {
            if t == max_t_asum {
                status.asum = problem.max_asum();
            } else {
                status.asum -= 2.0 * t * problem.sign(i);
            }
        }
        status.a[i] -= t;
        status.a[j] += t;
        status.value -= step.dvalue;
        for (idx, &k) in active_set.iter().enumerate() {
            status.ka[k] += t / problem.lambda() * (kj[idx] - ki[idx]);
        }
    });
}
