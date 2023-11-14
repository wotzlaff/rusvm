//! Kernels

mod cached;
pub use cached::CachedKernel;
mod precomputed;
pub use precomputed::PrecomputedKernel;
mod row;
use ndarray::ArrayView2;
pub use row::RowKernel;

/// An abstract kernel matrix
pub trait Kernel {
    /// Computes the ith row of the kernel matrix with entries according to `active_set` and saves it into the (preallocated) slice `ki`.
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &[usize]);

    /// Returns the size of the matrix (number of samples)
    fn size(&self) -> usize;

    /// Returns the ith diagonal element of the kernel matrix.
    fn diag(&self, i: usize) -> f64;

    /// Restricts (shrinks) the current active set (important for caching).
    fn restrict_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {}
    /// Expands (unshrinks) the current active set.
    fn set_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {}

    /// Computes a set of rows of the kernel matrix and hands them over to the callback `fun`.
    fn use_rows(&mut self, idxs: &[usize], active_set: &[usize], fun: &mut dyn FnMut(Vec<&[f64]>)) {
        let mut kidxs = Vec::with_capacity(idxs.len());
        let active_size = active_set.len();
        for &idx in idxs.iter() {
            let mut kidx = vec![0.0; active_size];
            self.compute_row(idx, &mut kidx, active_set);
            kidxs.push(kidx);
        }
        fun(kidxs.iter().map(|ki| ki.as_slice()).collect());
    }
}

/// Derivative of kernel matrix is available
pub trait Deriv {
    /// Computes the derivative of the ith row of the kernel matrix wrt hyperparameters.
    fn compute_row_deriv(&self, i: usize, dkis: &[&mut [f64]], active_set: &[usize]);

    /// Returns the number of hyperparameters.
    fn num_params(&self) -> usize;
}

/// Builds a RBF/Gaussian kernel matrix.
pub fn gaussian<'a>(arr: &'a ArrayView2<'a, f64>, gamma: f64) -> impl Kernel + 'a {
    let data = arr.outer_iter().collect();
    RowKernel::new(
        data,
        Box::new(move |&xi, &xj| {
            let dij = xi
                .iter()
                .zip(xj.iter())
                .fold(0.0, |acc, (xik, xjk)| acc + (xik - xjk).powi(2));
            (-gamma * dij).exp()
        }),
        Box::new(move |&_xi| 1.0),
    )
}

pub use gaussian as rbf;
