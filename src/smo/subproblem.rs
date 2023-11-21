use crate::problem::DualProblem;
use crate::status::Status;

pub fn newton(f: &dyn Fn(f64) -> (f64, f64, f64), x0: f64, xmax: f64) -> (f64, f64) {
    let mut x = x0;
    let (mut v, mut dv, mut ddv) = f(x);
    for _step in 0..5 {
        let dx_unc = if f64::is_finite(dv) { -dv / ddv } else { 1.0 };
        let dx = f64::min(dx_unc, xmax - x);
        if dv.abs() < 1e-6 || (dx != dx_unc && dv < 0.0) {
            break;
        }
        // println!("{step:5}: {v:10.4} {dx_unc:10.4} -> {dv:12.6}");
        let mut alpha = 1.0;
        let mut backstep = 0;
        loop {
            let x_n = x + alpha * dx;
            let (v_n, dv_n, ddv_n) = f(x_n);
            let dec = v_n - v;
            let dec_ref = alpha * dv * dx;
            // println!("> {backstep}: {dec} / {dec_ref} {dv_n}");
            if dec <= dec_ref || dec <= 0.0 {
                x = x_n;
                (v, dv, ddv) = (v_n, dv_n, ddv_n);
                break;
            }
            alpha *= 0.1;
            backstep += 1;
            assert!(backstep <= 20);
        }
    }
    (x, v)
}

#[derive(Debug)]
pub struct Step {
    pub t: f64,
    pub dvalue: f64,
}

#[derive(Debug)]
pub struct Subproblem {
    pub ij: (usize, usize),
    pub max_t: f64,
    pub q0: f64,
    pub p0: f64,
}

pub fn compute_step(problem: &dyn DualProblem, sprob: Subproblem, status: &Status) -> Step {
    let (i, j) = sprob.ij;
    let ai = status.a[i];
    let aj = status.a[j];
    if problem.is_quad() {
        let p = sprob.p0 + problem.d_dloss(i, ai) - problem.d_dloss(j, aj);
        let q = sprob.q0 + problem.d2_dloss(i, ai) + problem.d2_dloss(j, aj);
        let t = f64::min(p / f64::max(q, problem.regularization()), sprob.max_t);
        let dvalue = t * (0.5 * q * t - p);
        Step { t, dvalue }
    } else {
        let loss = problem.dloss(i, ai) + problem.dloss(j, aj);
        let (t, dvalue) = newton(
            &|t| {
                let v = t * (0.5 * sprob.q0 * t - sprob.p0) - loss
                    + problem.dloss(i, ai - t)
                    + problem.dloss(j, aj + t);
                let dv = sprob.q0 * t - sprob.p0 - problem.d_dloss(i, ai - t)
                    + problem.d_dloss(j, aj + t);
                let ddv: f64 = sprob.q0 + problem.d2_dloss(i, ai - t) + problem.d2_dloss(j, aj + t);
                (v, dv, ddv)
            },
            0.0,
            sprob.max_t,
        );
        Step { t, dvalue }
    }
}
