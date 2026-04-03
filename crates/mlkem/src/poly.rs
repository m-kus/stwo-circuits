use circuits::context::Context;
use circuits::ivalue::IValue;

use crate::constants::N;
use crate::ntt::{RqPoly, inv_ntt, ntt, ntt_pointwise_mul};
use crate::zq::{ZqWitness, zq_add, zq_sub};

#[cfg(test)]
#[path = "poly_test.rs"]
pub mod test;

/// Coefficient-wise addition of two polynomials.
/// Both inputs should have coefficients in [0, Q).
/// Output coefficients are unreduced, in [0, 2Q).
pub fn poly_add<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &RqPoly,
    b: &RqPoly,
) -> RqPoly {
    let coeffs = std::array::from_fn(|i| zq_add(ctx, a.coeffs[i], b.coeffs[i]));
    RqPoly { coeffs }
}

/// Coefficient-wise subtraction of two polynomials.
/// Both inputs should have coefficients in [0, Q).
/// Output coefficients are reduced to [0, Q).
pub fn poly_sub<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &RqPoly,
    b: &RqPoly,
) -> RqPoly {
    let coeffs = std::array::from_fn(|i| zq_sub(ctx, a.coeffs[i], b.coeffs[i]));
    RqPoly { coeffs }
}

/// Multiply two polynomials in NTT domain (both must already be NTT-transformed).
pub fn poly_pointwise_mul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &RqPoly,
    b: &RqPoly,
) -> RqPoly {
    ntt_pointwise_mul(ctx, a, b)
}

/// Full polynomial multiplication: NTT → pointwise → INTT.
/// Both inputs in coefficient domain, output in coefficient domain.
pub fn poly_mul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &RqPoly,
    b: &RqPoly,
) -> RqPoly {
    let mut a_ntt = a.clone();
    let mut b_ntt = b.clone();
    ntt(ctx, &mut a_ntt);
    ntt(ctx, &mut b_ntt);
    let mut result = ntt_pointwise_mul(ctx, &a_ntt, &b_ntt);
    inv_ntt(ctx, &mut result);
    result
}

/// Inner product of two polynomial vectors in NTT domain.
/// Both vectors must already be NTT-transformed.
/// Returns a single polynomial (sum of pointwise products).
pub fn inner_product_ntt<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &[RqPoly],
    b: &[RqPoly],
) -> RqPoly {
    assert_eq!(a.len(), b.len());
    assert!(!a.is_empty());

    let mut acc = ntt_pointwise_mul(ctx, &a[0], &b[0]);
    for i in 1..a.len() {
        let prod = ntt_pointwise_mul(ctx, &a[i], &b[i]);
        acc = poly_add(ctx, &acc, &prod);
    }
    acc
}

/// Matrix-vector multiplication: A * s where A is a k×k matrix of polynomials
/// and s is a k-vector. All in NTT domain.
/// Returns a k-vector of polynomials.
pub fn matrix_vec_mul_ntt<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a_matrix: &[Vec<RqPoly>],
    s_vec: &[RqPoly],
) -> Vec<RqPoly> {
    let k = s_vec.len();
    assert_eq!(a_matrix.len(), k);

    (0..k)
        .map(|i| {
            assert_eq!(a_matrix[i].len(), k);
            inner_product_ntt(ctx, &a_matrix[i], s_vec)
        })
        .collect()
}

/// Create a polynomial with all zero coefficients.
pub fn poly_zero<V: IValue>(ctx: &mut Context<V>) -> RqPoly {
    let zero = crate::zq::zq_constant(ctx, 0);
    RqPoly { coeffs: [zero; N] }
}
