use super::subproblem::{compute_step, Subproblem};
use crate::kernel::Kernel;
use crate::problem::DualProblem;
use crate::status::Status;

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
            Subproblem {
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
