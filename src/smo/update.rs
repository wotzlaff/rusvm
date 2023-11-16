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
    let pij: f64 = sprob.p0 + problem.d_dloss(i, ai) - problem.d_dloss(j, aj);
    let qij = sprob.q0 + problem.d2_dloss(i, ai) + problem.d2_dloss(j, aj);
    let t: f64 = if problem.is_quad() {
        f64::min(pij / f64::max(qij, problem.regularization()), sprob.max_t)
    } else {
        newton(
            &|t| {
                let v = t * (0.5 * sprob.q0 * t - sprob.p0)
                    + (problem.dloss(i, ai - t) - problem.dloss(i, ai))
                    + (problem.dloss(j, aj + t) - problem.dloss(j, aj));
                let dv = sprob.q0 * t - sprob.p0 - problem.d_dloss(i, ai - t)
                    + problem.d_dloss(j, aj + t);
                let ddv = sprob.q0 + problem.d2_dloss(i, ai - t) + problem.d2_dloss(j, aj + t);
                (v, f64::max(dv, -1e3), f64::min(ddv, 1e6))
            },
            0.0,
            sprob.max_t,
        )
    };
    let dvalue = if problem.is_quad() {
        t * (0.5 * qij * t - pij)
    } else {
        t * (0.5 * sprob.q0 * t - sprob.p0)
            + (problem.dloss(i, ai - t) - problem.dloss(i, ai))
            + (problem.dloss(j, aj + t) - problem.dloss(j, aj))
    };
    Step { t, dvalue }
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
        let max_tij = f64::min(status.a[i] - problem.lb(i), problem.ub(j) - status.a[j]);
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

        let mut t = step.t;
        if problem.sign(i) != problem.sign(j) {
            let rem_asum = problem.max_asum() - status.asum;
            if problem.sign(i) < 0.0 && 0.5 * rem_asum <= max_tij {
                t = 0.5 * rem_asum;
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
