use ndarray::ArrayView2;

/// A struct containing data for the computation of a kernel matrix with Gaussian kernel function.
pub struct GaussianKernel<'a> {
    gamma: f64,
    data: ArrayView2<'a, f64>,
    xsqr: Vec<f64>,
}

impl GaussianKernel<'_> {
    /// Creates a [`GaussianKernel`] for given scaling parameter `gamma` and feature matrix `data`.
    pub fn new(gamma: f64, data: ArrayView2<f64>) -> GaussianKernel {
        let &[n, nft] = data.shape() else {
            panic!("x has bad shape");
        };
        let mut xsqr = Vec::with_capacity(n);
        for i in 0..n {
            let mut xsqri = 0.0;
            for j in 0..nft {
                xsqri += data[[i, j]] * data[[i, j]];
            }
            xsqr.push(xsqri);
        }
        GaussianKernel { gamma, data, xsqr }
    }
}

impl super::Kernel for GaussianKernel<'_> {
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &Vec<usize>) {
        let xsqri = self.xsqr[i % self.xsqr.len()];
        let xi = self.data.row(i % self.xsqr.len());
        for (idx_j, &j) in active_set.iter().enumerate() {
            let xj = self.data.row(j % self.xsqr.len());
            let dij = xsqri + self.xsqr[j % self.xsqr.len()] - 2.0 * xi.dot(&xj);
            (*ki)[idx_j] = (-self.gamma * dij).exp();
        }
    }

    fn diag(&self, _i: usize) -> f64 {
        1.0
    }
}
