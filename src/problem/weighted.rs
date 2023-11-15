struct WeightedProblem<'a, T> {
    base: T,
    weights: &'a [f64],
}

impl WeightedProblem {
    // fn new(base:)
    fn weight(&self, i: usize) -> f64 {
        match self.weights {
            Some(weights) => weights[i],
            None => 1.0,
        }
    }

    /// Sets the weights of the individual samples.
    pub fn with_weights(mut self, weights: &'a [f64]) -> Self {
        self.weights = Some(weights);
        self
    }
}

// impl super::shrinking::ShrinkingBase for WeightedProblem<'_> {
//     fn lb(&self, i: usize) -> f64 {
//         if self.y[i] > 0.0 {
//             0.0
//         } else {
//             -self.weight(i)
//         }
//     }
//     fn ub(&self, i: usize) -> f64 {
//         if self.y[i] > 0.0 {
//             self.weight(i)
//         } else {
//             0.0
//         }
//     }
// }
