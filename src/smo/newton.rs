pub fn newton(f: &dyn Fn(f64) -> (f64, f64, f64), x0: f64, xmax: f64) -> (f64, f64) {
    let mut x = x0;
    let (mut v, mut dv, mut ddv) = f(x);
    for step in 0..5 {
        let dx_unc = -dv / ddv;
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
