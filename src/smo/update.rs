use super::status::Status;
use crate::kernel::Kernel;
use crate::problem::Problem;

pub fn update(
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
        let mut tij: f64 = f64::min(
            problem.lambda() * pij / f64::max(qij, problem.regularization()),
            max_tij,
        );
        if problem.sign(i) != problem.sign(j) {
            let rem_asum = problem.max_asum() - status.asum;
            if problem.sign(i) < 0.0 && rem_asum <= 2.0 * max_tij {
                tij = 0.5 * rem_asum;
                status.asum = problem.max_asum();
            } else {
                status.asum -= 2.0 * tij * problem.sign(i);
            }
        }
        status.a[i] -= tij;
        status.a[j] += tij;
        let tij_l = tij / problem.lambda();
        status.value -= tij * (0.5 * qij * tij_l - pij);
        for (idx, &k) in active_set.iter().enumerate() {
            status.ka[k] += tij_l * (kj[idx] - ki[idx]);
        }
    });
}
