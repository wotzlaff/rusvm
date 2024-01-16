use crate::Status;

pub struct ActiveSet {
    pub size: usize,
    pub size_positive: usize,
    pub positive: Vec<usize>,
    pub zeros: Vec<usize>,
}

impl ActiveSet {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            size_positive: 0,
            positive: Vec::new(),
            zeros: Vec::new(),
        }
    }

    pub fn all(&self) -> &[usize] {
        &self.positive[..]
    }

    pub fn positives(&self) -> &[usize] {
        &self.positive[..self.size_positive]
    }

    pub fn zeros(&self) -> &[usize] {
        &self.positive[self.size_positive..]
    }

    pub fn merge(&mut self) {
        self.size_positive = self.positive.len();
        self.positive.append(&mut self.zeros);
    }
}

/// Struct containing the elements (a, b, c)
pub struct Direction {
    /// Vector of coefficients
    pub a: Vec<f64>,
    /// Bias value
    pub b: f64,
    /// Shift value
    pub c: f64,
}

impl Direction {
    /// Creates a new direction struct
    pub fn new(size: usize) -> Self {
        Self {
            a: vec![0.0; size],
            b: 0.0,
            c: 0.0,
        }
    }
}

pub struct Sums {
    pub a: f64,
    pub g: f64,
    pub sa: f64,
    pub sg: f64,
    pub da_zeros: f64,
    pub sda_zeros: f64,
}

impl Sums {
    pub fn new() -> Self {
        Self {
            a: 0.0,
            g: 0.0,
            sa: 0.0,
            sg: 0.0,
            da_zeros: 0.0,
            sda_zeros: 0.0,
        }
    }
}

/// Status with additions (mainly second-order) information
pub struct StatusExtended {
    /// Underlying base status
    pub status: Status,
    /// Current direction type
    pub dir: Direction,
    /// Active set (containing nonzero 2nd loss derivatives and nonzero updates)
    pub active: ActiveSet,
    /// Some terms needed in Newton's method
    pub sums: Sums,
    /// Value of second loss derivative
    pub h: Vec<f64>,
    /// Place to store the current kernel matrix row
    pub ki: Vec<f64>,
}
