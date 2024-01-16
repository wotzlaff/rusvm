use super::Kernel;
use caches::{Cache, RawLRU};

/// A struct to cache rows of a kernel matrix.
pub struct CachedKernel<'a> {
    cache: RawLRU<usize, usize>,
    data: Vec<Vec<f64>>,
    base: Box<dyn Kernel + 'a>,
}

impl<'a> CachedKernel<'a> {
    /// Generates a cached version of the given kernel matrix.
    pub fn from(base: Box<dyn Kernel + 'a>, capacity: usize) -> CachedKernel<'a> {
        CachedKernel {
            cache: RawLRU::new(capacity).unwrap(),
            data: Vec::new(),
            base,
        }
    }
}

/// Add cache to base kernel
pub fn cache<'a>(base: Box<dyn Kernel + 'a>, cache_size: usize) -> Box<dyn Kernel + 'a> {
    if cache_size > 0 {
        Box::new(CachedKernel::from(base, cache_size))
    } else {
        base
    }
}

fn find_common<T>(a: &Vec<T>, b: &Vec<T>) -> Vec<usize>
where
    T: PartialEq,
{
    let mut res = Vec::new();
    let mut it: usize = 0;
    for (idx, i) in a.iter().enumerate() {
        if i == &b[it] {
            it += 1;
            res.push(idx);
            if it >= b.len() {
                break;
            }
        }
    }
    res
}

impl<'a> super::Kernel for CachedKernel<'a> {
    fn compute_row(&self, i: usize, ki: &mut [f64], active_set: &[usize]) {
        self.base.compute_row(i, ki, active_set);
    }

    fn size(&self) -> usize {
        self.base.size()
    }

    fn use_rows(&mut self, idxs: &[usize], active_set: &[usize], fun: &mut dyn FnMut(Vec<&[f64]>)) {
        assert!(idxs.len() <= self.cache.cap(), "cache size too small");
        let poss: Vec<_> = idxs
            .iter()
            .map(|&idx| match self.cache.get(&idx) {
                Some(&pos) => pos,
                None => {
                    let size = self.cache.cap();
                    let pos = if self.data.len() < size {
                        let ki: Vec<f64> = vec![0.0; active_set.len()];
                        let pos = self.data.len();
                        self.data.push(ki);
                        pos
                    } else {
                        let (_idx, pos) = self.cache.remove_lru().unwrap();
                        pos
                    };
                    self.base.compute_row(idx, &mut self.data[pos], active_set);
                    self.cache.put(idx, pos);
                    pos
                }
            })
            .collect();
        fun(poss
            .into_iter()
            .map(|pos| self.data[pos].as_slice())
            .collect());
    }

    fn restrict_active(&mut self, old: &Vec<usize>, new: &Vec<usize>) {
        let sub = find_common(old, new);
        for ki in self.data.iter_mut() {
            *ki = sub.iter().map(|&idx| ki[idx]).collect();
        }
    }

    fn set_active(&mut self, _old: &Vec<usize>, _new: &Vec<usize>) {
        self.cache = RawLRU::new(self.cache.cap()).unwrap();
        self.data = Vec::new();
    }

    fn diag(&self, i: usize) -> f64 {
        self.base.diag(i)
    }
}
