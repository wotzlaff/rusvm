//! Helper functions to obtain sensitivity information about the solution
use crate::kernel::Kernel;
use crate::newton::StatusExtended;
use crate::problem::PrimalProblem;

use rulinalg::matrix::decomposition::PartialPivLu;
use rulinalg::matrix::Matrix;
use rulinalg::vector::Vector;

/// Extended vector (a, b, c)
pub type EVec = (Vec<f64>, f64, f64);

/// Solves the transposed linearized system
pub fn solve_transposed_linearization(
    problem: &dyn PrimalProblem,
    kernel: &mut dyn Kernel,
    status_ext: &mut StatusExtended,
    rhs: &EVec,
) -> EVec {
    status_ext.active.merge();
    let n_active = status_ext.active.size_positive;
    if n_active == 0 {
        // TODO: fail
    }
    let mut signs = Vector::zeros(n_active);
    if problem.has_max_asum() {
        let mut sign_pos = false;
        let mut sign_neg = false;
        for (idx_i, &i) in status_ext.active.positives().iter().enumerate() {
            let si = problem.sign(i);
            sign_pos |= si > 0.0;
            sign_neg |= si < 0.0;
            signs[idx_i] = si;
        }
        if !(sign_pos && sign_neg) {
            // TODO: fail
        }
    }

    let rhs0 = Vector::from(rhs.0.clone());
    let mut sol = (vec![0.0; problem.size()], 0.0, 0.0);
    let full_set: Vec<_> = (0..problem.size()).collect();
    kernel.use_rows(status_ext.active.positives(), &full_set, &mut |kis| {
        let mut mat = Matrix::zeros(n_active, n_active);
        for (idx_i, &i) in status_ext.active.positives().iter().enumerate() {
            for (idx_j, &j) in status_ext.active.positives().iter().enumerate() {
                mat[[idx_i, idx_j]] = kis[idx_i][j] / problem.lambda();
                if idx_i == idx_j {
                    mat[[idx_i, idx_j]] += 1.0 / status_ext.h[i];
                }
            }
        }
        let mat_fact = PartialPivLu::decompose(mat).unwrap();

        let mat_inv_rhs = mat_fact.solve(rhs0.clone()).unwrap();
        let mat_inv_one = mat_fact.solve(Vector::ones(n_active)).unwrap();

        let vanishing: Vec<_> = (0..problem.size())
            .filter(|&i| status_ext.h[i] == 0.0)
            .collect();
        if problem.has_max_asum() {
            // solve system with two additional constraints
            let mat_inv_signs = mat_fact.solve(signs.clone()).unwrap();
            // create and solve 2x2 system
            let q00 = mat_inv_one.sum();
            let q01 = mat_inv_signs.sum();
            let q11 = mat_inv_signs.dot(&signs);
            let det = q00 * q11 - q01 * q01;
            let p0 = mat_inv_one.dot(&rhs0) - rhs.1;
            let p1 = mat_inv_signs.dot(&rhs0) - rhs.2;
            // extract solution for scalar variables
            let db = (q11 * p0 - q01 * p1) / det;
            sol.1 = db;
            let dc = (q00 * p1 - q01 * p0) / det;
            sol.2 = dc;
            for &j in vanishing.iter() {
                sol.0[j] = rhs0[j] - db - dc * problem.sign(j);
            }
        } else {
            // solve system with one additional constraints
            let db = (mat_inv_rhs.sum() - rhs.1) / mat_inv_one.sum();
            sol.1 = db;
            for &j in vanishing.iter() {
                sol.0[j] = rhs.0[j] - db;
            }
        };
        for (idx_i, &i) in status_ext.active.positives().iter().enumerate() {
            sol.0[i] = mat_inv_rhs[idx_i] / status_ext.h[i];
            for &j in vanishing.iter() {
                sol.0[j] -= kis[idx_i][j] / problem.lambda() * mat_inv_rhs[idx_i];
            }
        }
    });
    return sol;
}
