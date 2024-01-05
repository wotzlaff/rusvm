use crate::kernel::KernelFunction;
use crate::Status;

/// Evaluate the decision function for a particular sample.
pub fn predict<T>(
    elem: &T,
    data: &Vec<T>,
    status: &Status,
    lmbda: f64,
    kernel_function: &KernelFunction<T>,
) -> f64 {
    let mut v = 0.0;
    for (&ai, xi) in status.a.iter().zip(data) {
        if ai == 0.0 {
            continue;
        }
        let ki = kernel_function(xi, elem);
        v += ai * ki / lmbda;
    }
    v + status.b
}
