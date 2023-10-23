pub mod gaussian;
pub use gaussian::GaussianKernel;
pub mod cached;
pub use cached::CachedKernel;

pub trait Kernel {
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &Vec<usize>);
    fn diag(&self, i: usize) -> f64;

    fn restrict_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {}
    fn set_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {}

    fn use_rows(
        &mut self,
        idxs: Vec<usize>,
        active_set: &Vec<usize>,
        fun: &mut dyn FnMut(Vec<&[f64]>),
    ) {
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
