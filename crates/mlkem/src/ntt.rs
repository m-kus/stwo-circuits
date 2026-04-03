use circuits::context::Context;
use circuits::ivalue::IValue;

use crate::constants::{N, NTT_SCALE_INV, Q, ZETAS};
use crate::zq::{ZqVar, ZqWitness, mod_reduce_lazy, zq_add, zq_mul, zq_mul_const, zq_sub};

#[cfg(test)]
#[path = "ntt_test.rs"]
pub mod test;

/// A polynomial in R_q = Z_q[X]/(X^256 + 1), represented by 256 coefficients.
#[derive(Clone, Debug)]
pub struct RqPoly {
    pub coeffs: [ZqVar; 256],
}

/// NTT butterfly: given (a, b) and twiddle factor ζ, computes:
///   a' = a + ζ·b  (mod Q)
///   b' = a - ζ·b  (mod Q)
///
/// Both outputs are reduced to [0, Q) to prevent accumulation across layers.
/// Returns (a', b').
fn butterfly<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: ZqVar,
    b: ZqVar,
    zeta: u32,
) -> (ZqVar, ZqVar) {
    let t = zq_mul_const(ctx, b, zeta); // t ∈ [0, Q)
    // a ∈ [0, Q) (invariant maintained by reducing both outputs), t ∈ [0, Q)
    // sum ∈ [0, 2Q) < 2^13
    let sum = zq_add(ctx, a, t);
    let new_a = mod_reduce_lazy(ctx, sum.0, 13);
    let new_b = zq_sub(ctx, a, t); // zq_sub already reduces
    (new_a, new_b)
}

/// Forward NTT (Cooley-Tukey, decimation-in-time).
///
/// Follows the FIPS 203 NTT algorithm. After NTT, the polynomial is in
/// "NTT domain" where pointwise multiplication corresponds to ring multiplication.
pub fn ntt<V: IValue + ZqWitness>(ctx: &mut Context<V>, poly: &mut RqPoly) {
    let mut k = 1usize;
    let mut len = 128;
    while len >= 2 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k += 1;
            for j in start..(start + len) {
                let (a, b) = butterfly(ctx, poly.coeffs[j], poly.coeffs[j + len], zeta);
                poly.coeffs[j] = a;
                poly.coeffs[j + len] = b;
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

/// Inverse NTT butterfly: given (a, b) and twiddle factor ζ, computes:
///   a' = a + b
///   b' = ζ · (b - a)  (mod Q)
///
/// Uses the SAME zeta as forward NTT (not the inverse).
/// Both outputs are reduced to [0, Q).
/// Returns (a', b').
fn inv_butterfly<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: ZqVar,
    b: ZqVar,
    zeta: u32,
) -> (ZqVar, ZqVar) {
    // a, b ∈ [0, Q), sum ∈ [0, 2Q) < 2^13
    let sum = zq_add(ctx, a, b);
    let new_a = mod_reduce_lazy(ctx, sum.0, 13);
    let diff = zq_sub(ctx, b, a); // reduces internally
    let new_b = zq_mul_const(ctx, diff, zeta); // reduces internally
    (new_a, new_b)
}

/// Inverse NTT (Gentleman-Sande, decimation-in-frequency).
///
/// Follows the FIPS 203 / pq-crystals InvNTT. Uses the SAME zetas array as the
/// forward NTT but in reverse k order. The inverse comes from the reversed layer
/// structure and the different butterfly formula, not from inverting the twiddles.
pub fn inv_ntt<V: IValue + ZqWitness>(ctx: &mut Context<V>, poly: &mut RqPoly) {
    let mut k = 127usize;
    let mut len = 2;
    while len <= 128 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k = k.wrapping_sub(1);
            for j in start..(start + len) {
                let (a, b) = inv_butterfly(ctx, poly.coeffs[j], poly.coeffs[j + len], zeta);
                poly.coeffs[j] = a;
                poly.coeffs[j + len] = b;
            }
            start += 2 * len;
        }
        len <<= 1;
    }

    // Scale all coefficients by 128^{-1} mod Q
    for i in 0..N {
        poly.coeffs[i] = zq_mul_const(ctx, poly.coeffs[i], NTT_SCALE_INV);
    }
}

/// Pointwise multiplication in NTT domain.
///
/// In ML-KEM, the NTT maps to 128 degree-1 polynomial pairs, so pointwise
/// multiplication is actually 128 "basemul" operations (multiply two degree-1
/// polynomials modulo (X^2 - ζ^{2·brv(i)+1})).
pub fn ntt_pointwise_mul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &RqPoly,
    b: &RqPoly,
) -> RqPoly {
    let mut coeffs = a.coeffs;
    for i in 0..64 {
        let zeta = ZETAS[64 + i];
        let (r0, r1) = basemul(
            ctx,
            coeffs[4 * i],
            coeffs[4 * i + 1],
            b.coeffs[4 * i],
            b.coeffs[4 * i + 1],
            zeta,
        );
        coeffs[4 * i] = r0;
        coeffs[4 * i + 1] = r1;

        // Negative zeta for the second pair
        let neg_zeta = Q - ZETAS[64 + i];
        let (r2, r3) = basemul(
            ctx,
            coeffs[4 * i + 2],
            coeffs[4 * i + 3],
            b.coeffs[4 * i + 2],
            b.coeffs[4 * i + 3],
            neg_zeta,
        );
        coeffs[4 * i + 2] = r2;
        coeffs[4 * i + 3] = r3;
    }
    RqPoly { coeffs }
}

/// Base multiplication of two degree-1 polynomials modulo (X^2 - γ):
///   (a0 + a1·X)(b0 + b1·X) mod (X^2 - γ)
///   = (a0·b0 + a1·b1·γ) + (a0·b1 + a1·b0)·X
///
/// Returns (result_0, result_1).
fn basemul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a0: ZqVar,
    a1: ZqVar,
    b0: ZqVar,
    b1: ZqVar,
    gamma: u32,
) -> (ZqVar, ZqVar) {
    let a0_b0 = zq_mul(ctx, a0, b0);
    let a1_b1 = zq_mul(ctx, a1, b1);
    let a0_b1 = zq_mul(ctx, a0, b1);
    let a1_b0 = zq_mul(ctx, a1, b0);

    let a1_b1_gamma = zq_mul_const(ctx, a1_b1, gamma);
    let r0 = zq_add(ctx, a0_b0, a1_b1_gamma);
    let r1 = zq_add(ctx, a0_b1, a1_b0);
    (r0, r1)
}

/// Convenience: compute polynomial multiplication via NTT.
/// Both inputs must be in coefficient domain.
/// Returns result in coefficient domain.
pub fn poly_mul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &mut RqPoly,
    b: &mut RqPoly,
) -> RqPoly {
    ntt(ctx, a);
    ntt(ctx, b);
    let mut result = ntt_pointwise_mul(ctx, a, b);
    inv_ntt(ctx, &mut result);
    result
}
