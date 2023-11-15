//! Primal training problem
use super::base::ProblemBase;
use crate::status::Status;

/// Base for the definition of a primal training problem
pub trait PrimalProblem: ProblemBase {
    /// Computes the primal and dual objective function values.
    fn objective(&self, status: &Status) -> f64 {
        let mut reg = 0.0;
        let mut loss_primal = 0.0;
        for i in 0..self.size() {
            // compute regularization
            reg += status.ka[i] * status.a[i];
            // compute primal loss
            let ti = status.ka[i] + status.b + self.sign(i) * status.c;
            loss_primal += self.loss(i, ti);
        }
        let asum_term = if self.max_asum() < f64::INFINITY {
            self.max_asum() * status.c
        } else {
            0.0
        };
        0.5 * reg + loss_primal + asum_term
    }

    /// Computes the ith loss function.
    fn loss(&self, i: usize, ti: f64) -> f64;
    /// Computes the first derivative of the ith loss function.
    fn d_loss(&self, i: usize, ti: f64) -> f64;
    /// Computes the second derivative of the ith loss function.
    fn d2_loss(&self, i: usize, ti: f64) -> f64;
}

/// Base for the definition of a primal training problem
pub trait PrimalLabelProblem: super::base::LabelProblem {
    /// Computes the loss function with label yi.
    fn label_loss(&self, i: usize, ti: f64, yi: Self::T) -> f64;
    /// Computes the first derivative of the loss function with label yi.
    fn d_label_loss(&self, i: usize, ti: f64, yi: Self::T) -> f64;
    /// Computes the second derivative of the loss function with label yi.
    fn d2_label_loss(&self, i: usize, ti: f64, yi: Self::T) -> f64;
}

// impl ProblemBase for dyn PrimalProblem {}

impl<P> PrimalProblem for P
where
    P: PrimalLabelProblem,
{
    /// Computes the ith loss function.
    fn loss(&self, i: usize, ti: f64) -> f64 {
        self.label_loss(i, ti, self.label(i))
    }
    /// Computes the first derivative of the ith loss function.
    fn d_loss(&self, i: usize, ti: f64) -> f64 {
        self.d_label_loss(i, ti, self.label(i))
    }
    /// Computes the second derivative of the ith loss function.
    fn d2_loss(&self, i: usize, ti: f64) -> f64 {
        self.d2_label_loss(i, ti, self.label(i))
    }
}
