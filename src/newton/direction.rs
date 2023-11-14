use super::status_extended::StatusExtended;
use crate::kernel::Kernel;
use crate::problem::Problem;
use ndarray::prelude::*;
use ndarray_linalg::{FactorizeInto, Solve};

pub enum DirectionType {
    Gradient,
    Newton,
    NoStep,
}

pub fn gradient(problem: &dyn Problem, _kernel: &mut dyn Kernel, status_ext: &mut StatusExtended) {
    for i in 0..problem.size() {
        status_ext.dir.a[i] = status_ext.status.a[i] + status_ext.status.g[i];
    }
    status_ext.dir.b = status_ext.sums.g / problem.lambda();
}

fn compute_matrix_and_rhs(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    status_ext: &mut StatusExtended,
) -> (Array2<f64>, Array1<f64>) {
    let h: &Vec<f64> = &status_ext.h;
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
    (mat, rhs)
}

pub fn newton_with_fallback(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    status_ext: &mut StatusExtended,
) -> DirectionType {
    status_ext.active.merge();
    let n_active = status_ext.active.size_positive;
    if n_active == 0 {
        gradient(problem, kernel, status_ext);
        return DirectionType::Gradient;
    }
    let mut signs = Array::zeros((n_active,));
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
            gradient(problem, kernel, status_ext);
            return DirectionType::Gradient;
        }
    }
    let (mat, rhs) = compute_matrix_and_rhs(problem, kernel, status_ext);
    let mat_fact = mat.factorize_into().unwrap();
    let mat_inv_rhs = mat_fact.solve(&rhs).unwrap();
    let mat_inv_one = mat_fact.solve_into(Array::ones((n_active,))).unwrap();

    let sums = &status_ext.sums;
    let rhs_b = sums.a - sums.da_zeros;
    let da_nonzero = if problem.has_max_asum() {
        // solve system with two additional constraints
        let rhs_c = sums.sa - problem.max_asum() - sums.sda_zeros;
        let mat_inv_signs = mat_fact.solve(&signs).unwrap();
        // create and solve 2x2 system
        let q00 = mat_inv_one.sum();
        let q01 = mat_inv_signs.sum();
        let q11 = mat_inv_signs.dot(&signs);
        let det = q00 * q11 - q01 * q01;
        let p0 = mat_inv_one.dot(&rhs) - rhs_b;
        let p1 = mat_inv_signs.dot(&rhs) - rhs_c;
        // extract solution for scalar variables
        let db = (q11 * p0 - q01 * p1) / det;
        status_ext.dir.b = db;
        let dc = (q00 * p1 - q01 * p0) / det;
        status_ext.dir.c = dc;
        mat_inv_rhs - db * mat_inv_one - dc * mat_inv_signs
    } else {
        // solve system with one additional constraints
        let db = (mat_inv_rhs.sum() - rhs_b) / mat_inv_one.sum();
        status_ext.dir.b = db;
        mat_inv_rhs - db * mat_inv_one
    };
    for (idx_i, &i) in status_ext.active.positives().iter().enumerate() {
        status_ext.dir.a[i] = da_nonzero[idx_i];
    }
    return DirectionType::Newton;
}
