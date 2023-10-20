use numpy::ndarray::ArrayView2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::conversion::IntoPy;
use pyo3::prelude::*;
use pyo3::types::PyDict;
pub struct State {
    a: Vec<f64>,
    b: f64,
    violation: f64,
    value: f64,
    ka: Vec<f64>,
    g: Vec<f64>,
    active_set: Vec<usize>,
}

pub trait Problem {
    fn quad(&self, state: &State, i: usize) -> f64;
    fn grad(&self, state: &State, i: usize) -> f64;
    fn size(&self) -> usize;
    fn lb(&self, i: usize) -> f64;
    fn ub(&self, i: usize) -> f64;
    fn sign(&self, i: usize) -> f64;
    fn is_optimal(&self, state: &State, tol: f64) -> bool;

    fn get_lambda(&self) -> f64;
    fn get_regularization(&self) -> f64;
}

pub trait Kernel {
    fn use_rows(&self, idxs: Vec<usize>, fun: &mut dyn FnMut(Vec<&[f64]>));
    fn diag(&self, i: usize) -> f64;
}

struct GaussianKernel<'a> {
    gamma: f64,
    data: ArrayView2<'a, f64>,
    xsqr: Vec<f64>,
}

impl<'a> GaussianKernel<'a> {
    fn new(gamma: f64, data: ArrayView2<'a, f64>) -> GaussianKernel<'a> {
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
    fn compute_row(&self, i: usize, ki: &mut [f64]) {
        let &[n, _nft] = self.data.shape() else {
            panic!("x has bad shape");
        };
        let xsqri = self.xsqr[i];
        let xi = self.data.row(i);
        for j in 0..n {
            let xj = self.data.row(j);
            let dij = xsqri + self.xsqr[j] - 2.0 * xi.dot(&xj);
            (*ki)[j] = (-self.gamma * dij).exp();
        }
    }
}

impl Kernel for GaussianKernel<'_> {
    fn use_rows(&self, idxs: Vec<usize>, fun: &mut dyn FnMut(Vec<&[f64]>)) {
        let mut kidxs = Vec::with_capacity(2);
        for &idx in &idxs {
            // TODO: active set
            let mut kidx = vec![0.0; self.data.shape()[0]];
            self.compute_row(idx, &mut kidx);
            kidxs.push(kidx);
        }
        fun(kidxs.iter().map(|ki| ki.as_slice()).collect());
    }

    fn diag(&self, _i: usize) -> f64 {
        1.0
    }
}

pub struct Classification {
    smoothing: f64,
    lambda: f64,
    shift: f64,
    y: Vec<f64>,
    w: Vec<f64>,
}

impl Problem for Classification {
    fn quad(&self, _state: &State, i: usize) -> f64 {
        2.0 * self.smoothing * self.lambda / self.w[i]
    }
    fn grad(&self, state: &State, i: usize) -> f64 {
        state.ka[i] - self.shift * self.y[i]
            + self.smoothing * self.y[i] * (2.0 * self.y[i] * state.a[i] / self.w[i] - 1.0)
    }
    fn size(&self) -> usize {
        self.y.len()
    }
    fn lb(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            0.0
        } else {
            -self.w[i]
        }
    }
    fn ub(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            self.w[i]
        } else {
            0.0
        }
    }
    fn sign(&self, i: usize) -> f64 {
        if self.y[i] > 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    fn is_optimal(&self, state: &State, tol: f64) -> bool {
        self.lambda * state.violation < tol
    }

    fn get_lambda(&self) -> f64 {
        self.lambda
    }
    fn get_regularization(&self) -> f64 {
        1e-12
    }
}

fn find_mvp_signed(
    problem: &impl Problem,
    state: &mut State,
    sign: f64,
) -> (f64, f64, usize, usize) {
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut idx_i: usize = 0;
    let mut idx_j: usize = 0;
    for (idx, &i) in state.active_set.iter().enumerate() {
        let g_i = problem.grad(state, i);
        state.g[i] = g_i;
        if problem.sign(i) * sign >= 0.0 {
            if state.a[i] > problem.lb(i) && g_i > g_max {
                idx_i = idx;
                g_max = g_i;
            }
            if state.a[i] < problem.ub(i) && g_i < g_min {
                idx_j = idx;
                g_min = g_i;
            }
        }
    }
    (g_max - g_min, g_max + g_min, idx_i, idx_j)
}

fn find_mvp(problem: &impl Problem, state: &mut State) -> (usize, usize) {
    let (dij, sij, idx_i, idx_j) = find_mvp_signed(problem, state, 0.0);
    state.b = -0.5 * sij;
    state.violation = dij;
    (idx_i, idx_j)
}

fn descent(q: f64, p: f64, t_max0: f64, t_max1: f64, lmbda: f64, regularization: f64) -> f64 {
    if p <= 0.0 || t_max1 == 0.0 {
        0.0
    } else {
        let t = f64::min(
            lmbda * p / f64::max(q, regularization),
            f64::min(t_max0, t_max1),
        );
        t * (p - 0.5 / lmbda * q * t)
    }
}

fn find_ws2(
    problem: &impl Problem,
    kernel: &impl Kernel,
    idx_i0: usize,
    idx_j1: usize,
    state: &State,
    sign: f64,
) -> (usize, usize) {
    let i0 = state.active_set[idx_i0];
    let j1 = state.active_set[idx_j1];
    let gi0 = state.g[i0];
    let gj1 = state.g[j1];
    let mut max_d0 = 0.0;
    let mut max_d1 = 0.0;
    let mut idx_j0 = idx_j1;
    let mut idx_i1 = idx_i0;
    kernel.use_rows([i0, j1].to_vec(), &mut |kij: Vec<&[f64]>| {
        let ki0 = kij[0];
        let kj1 = kij[1];
        let ki0i0 = ki0[idx_i0];
        let kj1j1 = kj1[idx_j1];
        let max_ti0 = state.a[i0] - problem.lb(i0);
        let max_tj1 = problem.ub(j1) - state.a[j1];

        for (idx_r, &r) in state.active_set.iter().enumerate() {
            if sign * problem.sign(r) < 0.0 {
                continue;
            }
            let gr = state.g[r];
            let krr = kernel.diag(r);
            let pi0r = gi0 - gr;
            let prj1 = gr - gj1;
            let d_upr = problem.ub(r) - state.a[r];
            let d_dnr = state.a[r] - problem.lb(r);

            if d_upr > 0.0 && pi0r > 0.0 {
                let qi0 = ki0i0 + krr - 2.0 * ki0[idx_r]
                    + problem.quad(state, i0)
                    + problem.quad(state, r);
                let di0r = descent(
                    qi0,
                    pi0r,
                    max_ti0,
                    d_upr,
                    problem.get_lambda(),
                    problem.get_regularization(),
                );
                if di0r > max_d0 {
                    idx_j0 = idx_r;
                    max_d0 = di0r;
                }
            }

            if d_dnr > 0.0 && prj1 > 0.0 {
                let qj1 = kj1j1 + krr - 2.0 * kj1[idx_r]
                    + problem.quad(state, j1)
                    + problem.quad(state, r);
                let drj1 = descent(
                    qj1,
                    prj1,
                    max_tj1,
                    d_dnr,
                    problem.get_lambda(),
                    problem.get_regularization(),
                );
                if drj1 > max_d1 {
                    idx_i1 = idx_r;
                    max_d1 = drj1;
                }
            }
        }
    });
    if max_d0 > max_d1 {
        (idx_i0, idx_j0)
    } else {
        (idx_i1, idx_j1)
    }
}

pub struct SMOResult {
    a: Vec<f64>,
    b: f64,
    value: f64,
    violation: f64,
    steps: usize,
}

impl SMOResult {
    fn from_state(state: &State) -> SMOResult {
        SMOResult {
            a: state.a.to_vec(),
            b: state.b,
            value: state.value,
            violation: state.violation,
            steps: 0,
        }
    }
}

impl IntoPy<PyObject> for SMOResult {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let res = PyDict::new(py);
        let _ = res.set_item("a", self.a);
        let _ = res.set_item("b", self.b);
        let _ = res.set_item("value", self.value);
        let _ = res.set_item("violation", self.violation);
        let _ = res.set_item("steps", self.steps);
        res.into_py(py)
    }
}

fn update(
    problem: &impl Problem,
    kernel: &impl Kernel,
    idx_i: usize,
    idx_j: usize,
    state: &mut State,
) {
    let i = state.active_set[idx_i];
    let j = state.active_set[idx_j];
    kernel.use_rows([i, j].to_vec(), &mut |kij: Vec<&[f64]>| {
        let ki = kij[0];
        let kj = kij[1];
        let pij = state.g[i] - state.g[j];
        let qij = ki[idx_i] + kj[idx_j] - 2.0 * ki[idx_j]
            + problem.quad(state, i)
            + problem.quad(state, j);
        let max_tij = f64::min(state.a[i] - problem.lb(i), problem.ub(j) - state.a[j]);
        let tij: f64 = f64::min(
            problem.get_lambda() * pij / f64::max(qij, problem.get_regularization()),
            max_tij,
        );
        state.a[i] -= tij;
        state.a[j] += tij;
        let tij_l = tij / problem.get_lambda();
        state.value -= tij * (0.5 * qij * tij_l - pij);
        for (idx, &k) in state.active_set.iter().enumerate() {
            state.ka[k] += tij_l * (kj[idx] - ki[idx]);
        }
    });
}

pub fn solve(
    problem: &impl Problem,
    kernel: &impl Kernel,
    tol: f64,
    max_steps: usize,
    verbose: usize,
    second_order: bool,
) -> SMOResult {
    let n = problem.size();
    let mut state = State {
        a: vec![0.0; n],
        b: 0.0,
        violation: f64::INFINITY,
        value: 0.0,
        ka: vec![0.0; n],
        g: vec![0.0; n],
        active_set: (0..n).collect(),
    };

    let mut step: usize = 0;
    loop {
        // TODO: check time limit
        if step >= max_steps {
            break;
        }
        // TODO: callback
        // TODO: shrinking
        let (idx_i0, idx_j1) = find_mvp(problem, &mut state);
        let optimal = problem.is_optimal(&state, tol);

        if verbose > 0 && (step % verbose == 0 || optimal) {
            println!("{:10} {:10.6} {:10.6}", step, state.violation, state.value)
        }

        if optimal {
            // TOOD: check shrinking
            break;
        }

        let (idx_i, idx_j) = if second_order {
            find_ws2(problem, kernel, idx_i0, idx_j1, &state, 0.0)
        } else {
            (idx_i0, idx_j1)
        };
        update(problem, kernel, idx_i, idx_j, &mut state);
        step += 1;
    }
    let mut result = SMOResult::from_state(&state);
    result.steps = step;
    result
}

#[pymodule]
fn smorust<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(signature = (x, y, lmbda = 1e-3, smoothing = 0.0, tol = 1e-4, max_steps = 1_000_000_000, verbose = 0, second_order = true))]
    fn solve_classification<'py>(
        _py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        lmbda: f64,
        smoothing: f64,
        tol: f64,
        max_steps: usize,
        verbose: usize,
        second_order: bool,
    ) -> PyResult<SMOResult> {
        let n = y.len();
        let problem = Classification {
            smoothing,
            lambda: lmbda,
            shift: 1.0,
            y: y.to_vec()?,
            w: vec![1.0; n],
        };
        let data = x.as_array();
        let kernel = GaussianKernel::new(1.0, data);
        let result = solve(&problem, &kernel, tol, max_steps, verbose, second_order);
        Ok(result)
    }
    Ok(())
}
