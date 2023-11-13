use super::Kernel;
use ndarray::Array2;

/// A struct to precompute rows of a kernel matrix.
pub struct PrecomputedKernel {
    n: usize,
    kernel_matrix: Array2<f64>,
}

impl PrecomputedKernel {
    /// Generates a precomputed version of the given kernel matrix.
    pub fn from(base: &impl Kernel) -> Self {
        let n = base.size();
        let mut kernel_matrix = ndarray::Array::zeros((n, n));
        for i in 0..n {
            let mut row_i = kernel_matrix.row_mut(i);
            let ki = row_i.as_slice_mut().unwrap();
            base.compute_row(i, ki, Vec::from_iter(0..=i).as_slice());
            for j in 0..i {
                kernel_matrix[(j, i)] = kernel_matrix[(i, j)];
            }
        }
        PrecomputedKernel { n, kernel_matrix }
    }
}

impl super::Kernel for PrecomputedKernel {
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &[usize]) {
        let n = self.size();
        for (idx_j, &j) in active_set.iter().enumerate() {
            ki[idx_j] = self.kernel_matrix[(i % n, j % n)]
        }
    }

    fn size(&self) -> usize {
        self.n
    }

    fn diag(&self, i: usize) -> f64 {
        let n = self.size();
        self.kernel_matrix[(i % n, i % n)]
    }
}
