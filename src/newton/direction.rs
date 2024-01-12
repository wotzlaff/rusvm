pub enum DirectionType {
    Gradient,
    Newton,
    NoStep,
}

#[cfg(feature = "lapack")]
mod direction_lapack;
use direction_lapack::{compute_matrix_and_rhs, newton_with_fallback};

#[cfg(not(feature = "lapack"))]
mod direction_nolapack;
use direction_nolapack::{compute_matrix_and_rhs, newton_with_fallback};
