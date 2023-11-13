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

pub fn newton_with_fallback(
    problem: &dyn Problem,
    kernel: &mut dyn Kernel,
    status_ext: &mut StatusExtended,
) -> DirectionType {
    if status_ext.active.positive.len() == 0 {
        gradient(problem, kernel, status_ext);
        return DirectionType::Gradient;
    }
    status_ext.active.merge();
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
    return DirectionType::Newton;
}
