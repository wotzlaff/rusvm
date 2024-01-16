//! Gaussian kernel matrix
use super::{Kernel, RowKernel};
use ndarray::ArrayView2;

/// Computes simple Gaussian kernel function.
pub fn kernel(xi: &[f64], xj: &[f64], gamma: f64) -> f64 {
    let dij = xi
        .iter()
        .zip(xj.iter())
        .fold(0.0, |acc, (xik, xjk)| acc + (xik - xjk).powi(2));
    (-gamma * dij).exp()
}

/// Builds a Gaussian kernel matrix.
pub fn from_array<'a>(arr: &'a ArrayView2<'a, f64>, gamma: f64) -> impl Kernel + 'a {
    let data = arr.outer_iter().collect();
    RowKernel::new(
        data,
        Box::new(move |&xi, &xj| kernel(xi.as_slice().unwrap(), xj.as_slice().unwrap(), gamma)),
        Box::new(move |&_xi| 1.0),
    )
}

/// Builds a Gaussian kernel matrix.
pub fn from_vecs<'a>(data: Vec<&'a [f64]>, gamma: f64) -> impl Kernel + 'a {
    RowKernel::new(
        data,
        Box::new(move |xi: &&'a [f64], xj: &&'a [f64]| kernel(xi, xj, gamma)),
        Box::new(move |&_xi| 1.0),
    )
}
