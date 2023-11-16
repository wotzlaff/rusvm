//! Dual training problem
use super::base::ProblemBase;
use super::shrinking::ShrinkingBase;
use crate::status::Status;

/// Base for training problem definition
pub trait DualProblem: ProblemBase + ShrinkingBase {
    /// Computes the primal and dual objective function values.
    fn objective(&self, status: &Status) -> f64 {
        let mut reg = 0.0;
        let mut loss_dual = 0.0;
        for i in 0..self.size() {
            // compute regularization
            reg += status.ka[i] * status.a[i];
            // compute dual loss
            loss_dual += self.dloss(i, status.a[i]);
        }
        0.5 * reg + loss_dual
    }
    /// Determines whether the problem is quadratic.
    fn is_quad(&self) -> bool {
        false
    }

    /// Computes the ith dual loss function.
    fn dloss(&self, i: usize, ai: f64) -> f64;
    /// Computes the first derivative of the ith dual loss function.
    fn d_dloss(&self, i: usize, ai: f64) -> f64;
    /// Computes the second derivative of the ith dual loss function.
    fn d2_dloss(&self, i: usize, ai: f64) -> f64;
}

/// Base for the definition of a dual training problem
pub trait DualLabelProblem: super::base::LabelProblem {
    /// Computes the dual loss function with label yi.
    fn label_dloss(&self, i: usize, ai: f64, yi: Self::T) -> f64;
    /// Computes the first derivative of the dual loss function with label yi.
    fn d_label_dloss(&self, i: usize, ai: f64, yi: Self::T) -> f64;
    /// Computes the second derivative of the dual loss function with label yi.
    fn d2_label_dloss(&self, i: usize, ai: f64, yi: Self::T) -> f64;
    /// Determines whether the dual loss function is quadratic.
    fn is_quad(&self) -> bool {
        false
    }
}

impl<P> DualProblem for P
where
    P: DualLabelProblem + ShrinkingBase,
{
    fn dloss(&self, i: usize, ai: f64) -> f64 {
        self.label_dloss(i, ai, self.label(i))
    }
    fn d_dloss(&self, i: usize, ai: f64) -> f64 {
        self.d_label_dloss(i, ai, self.label(i))
    }
    fn d2_dloss(&self, i: usize, ai: f64) -> f64 {
        self.d2_label_dloss(i, ai, self.label(i))
    }
    fn is_quad(&self) -> bool {
        self.is_quad()
    }
}
