/// Function type for kernel functions
pub type KernelFunction<T> = Box<dyn Fn(&T, &T) -> f64>;
/// Function type for diagonal entries of kernel functions
pub type DiagFunction<T> = Box<dyn Fn(&T) -> f64>;

/// A struct containing data for the computation of a kernel matrix with kernel function.
pub struct RowKernel<T> {
    data: Vec<T>,
    kernel_function: KernelFunction<T>,
    diag_function: DiagFunction<T>,
}

impl<T> RowKernel<T> {
    /// Creates a [`RowKernel`] for given feature matrix and kernel function.
    pub fn new(
        data: Vec<T>,
        kernel_function: KernelFunction<T>,
        diag_function: DiagFunction<T>,
    ) -> Self {
        RowKernel {
            data,
            kernel_function,
            diag_function,
        }
    }
}

impl<T> super::Kernel for RowKernel<T> {
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &[usize]) {
        let xi = &self.data[i % self.data.len()];
        for (idx_j, &j) in active_set.iter().enumerate() {
            let xj = &self.data[j % self.data.len()];
            (*ki)[idx_j] = (self.kernel_function)(&xi, &xj);
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn diag(&self, i: usize) -> f64 {
        let xi = &self.data[i % self.data.len()];
        (self.diag_function)(xi)
    }
}
