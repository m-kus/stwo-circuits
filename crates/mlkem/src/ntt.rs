use circuits::context::{Context, Var};
use circuits::ivalue::IValue;
use circuits::ops::pointwise_mul;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::constants::{N, NTT_SCALE_INV, Q, ZETAS};
use crate::zq::{
    ZqVar, ZqWitness, mod_reduce_lazy, mod_reduce_lazy_batch4, zq_add, zq_mul_const,
    zq_mul_const_ex, zq_sub_ex,
};

#[cfg(test)]
#[path = "ntt_test.rs"]
pub mod test;

/// A polynomial in R_q = Z_q[X]/(X^256 + 1), represented by 256 coefficients.
#[derive(Clone, Debug)]
pub struct RqPoly {
    pub coeffs: [ZqVar; 256],
}

/// Compute the number of bits needed to represent values up to `max_val`.
const fn bits_for(max_val: u32) -> u32 {
    if max_val == 0 { 1 } else { 32 - max_val.leading_zeros() }
}

/// Forward NTT butterfly WITHOUT reducing the addition output.
///
/// a can be up to `a_max`, b can be up to `b_max`.
/// Returns (a + t, a - t) where t = zeta * b mod Q.
/// new_a is unreduced (up to a_max + Q), new_b is reduced to [0, Q).
fn butterfly_lazy<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: ZqVar,
    b: ZqVar,
    zeta: u32,
    b_max: u32,
    a_max: u32,
) -> (ZqVar, ZqVar) {
    // t = b * zeta, reduced to [0, Q)
    let mul_input_bits = bits_for(b_max * Q);
    let t = zq_mul_const_ex(ctx, b, zeta, mul_input_bits);

    // new_a = a + t (UNREDUCED — saves ~100 gates per butterfly)
    let new_a = zq_add(ctx, a, t);

    // new_b = a - t (reduced)
    // offset must be >= a_max to prevent underflow
    let offset = (a_max / Q + 1) * Q; // round up to multiple of Q
    let sub_input_bits = bits_for(a_max + offset);
    let new_b = zq_sub_ex(ctx, a, t, offset, sub_input_bits);

    (new_a, new_b)
}

/// Process 4 forward NTT butterflies with batched range checks.
///
/// Takes 4 (a, b) pairs that share the same zeta and bounds.
/// Returns 4 (new_a, new_b) pairs. new_a is unreduced, new_b is reduced.
fn butterfly_batch4<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: [ZqVar; 4],
    b: [ZqVar; 4],
    zeta: u32,
    b_max: u32,
    a_max: u32,
) -> ([ZqVar; 4], [ZqVar; 4]) {
    let zeta_var = ctx.constant(QM31::from(M31::from(zeta)));
    let offset = (a_max / Q + 1) * Q;
    let offset_const = ctx.constant(QM31::from(M31::from(offset)));

    // Step 1: Compute 4 products b[i] * zeta (no reduction yet)
    let prods: [Var; 4] = std::array::from_fn(|i| pointwise_mul(ctx, b[i].0, zeta_var));

    // Step 2: Batch-reduce the 4 products
    let mul_input_bits = bits_for(b_max * Q);
    let t = mod_reduce_lazy_batch4(ctx, prods, mul_input_bits);

    // Step 3: Compute 4 unreduced sums (new_a = a + t) and 4 diffs for reduction
    let new_a: [ZqVar; 4] = std::array::from_fn(|i| zq_add(ctx, a[i], t[i]));

    let diffs: [Var; 4] = std::array::from_fn(|i| {
        circuits::eval!(ctx, ((a[i].0) + (offset_const)) - (t[i].0))
    });

    // Step 4: Batch-reduce the 4 diffs
    let sub_input_bits = bits_for(a_max + offset);
    let new_b = mod_reduce_lazy_batch4(ctx, diffs, sub_input_bits);

    (new_a, new_b)
}

/// Forward NTT (Cooley-Tukey, decimation-in-time).
///
/// Optimized: skips add-reduce in butterflies and uses Simd(4) batch range checks.
pub fn ntt<V: IValue + ZqWitness>(ctx: &mut Context<V>, poly: &mut RqPoly) {
    let mut k = 1usize;
    let mut len = 128;

    // Track max coefficient value. Inputs assumed in [0, Q).
    let mut max_coeff = Q;

    while len >= 2 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k += 1;

            if len >= 4 {
                // Process butterflies in batches of 4
                let mut j = start;
                while j + 3 < start + len {
                    let a4: [ZqVar; 4] =
                        std::array::from_fn(|i| poly.coeffs[j + i]);
                    let b4: [ZqVar; 4] =
                        std::array::from_fn(|i| poly.coeffs[j + i + len]);

                    let (new_a4, new_b4) =
                        butterfly_batch4(ctx, a4, b4, zeta, max_coeff, max_coeff);

                    for i in 0..4 {
                        poly.coeffs[j + i] = new_a4[i];
                        poly.coeffs[j + i + len] = new_b4[i];
                    }
                    j += 4;
                }
                // Handle remaining butterflies (when len % 4 != 0)
                while j < start + len {
                    let (a, b) = butterfly_lazy(
                        ctx,
                        poly.coeffs[j],
                        poly.coeffs[j + len],
                        zeta,
                        max_coeff,
                        max_coeff,
                    );
                    poly.coeffs[j] = a;
                    poly.coeffs[j + len] = b;
                    j += 1;
                }
            } else {
                // len < 4: process individually
                for j in start..(start + len) {
                    let (a, b) = butterfly_lazy(
                        ctx,
                        poly.coeffs[j],
                        poly.coeffs[j + len],
                        zeta,
                        max_coeff,
                        max_coeff,
                    );
                    poly.coeffs[j] = a;
                    poly.coeffs[j + len] = b;
                }
            }
            start += 2 * len;
        }
        // After each layer: a' values are unreduced (max + Q), b' values are reduced (< Q)
        // Max for next layer = max(prev_max + Q, Q) = prev_max + Q
        max_coeff += Q;
        len >>= 1;
    }
}

/// Inverse NTT butterfly with reduced addition output.
///
/// Unlike forward butterfly, both a and b can be unreduced, so new_a = a + b
/// would grow exponentially. We MUST reduce it.
fn inv_butterfly_lazy<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: ZqVar,
    b: ZqVar,
    zeta: u32,
    max_coeff: u32,
) -> (ZqVar, ZqVar) {
    // new_a = a + b, reduced
    let sum = zq_add(ctx, a, b);
    let sum_bits = bits_for(2 * max_coeff);
    let new_a = mod_reduce_lazy(ctx, sum.0, sum_bits);

    // new_b = zeta * (b - a), reduced
    let offset = (max_coeff / Q + 1) * Q;
    let sub_bits = bits_for(max_coeff + offset);
    let diff = zq_sub_ex(ctx, b, a, offset, sub_bits);
    let new_b = zq_mul_const(ctx, diff, zeta);

    (new_a, new_b)
}

/// Process 4 inverse NTT butterflies with batched range checks.
fn inv_butterfly_batch4<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: [ZqVar; 4],
    b: [ZqVar; 4],
    zeta: u32,
    max_coeff: u32,
) -> ([ZqVar; 4], [ZqVar; 4]) {
    let zeta_var = ctx.constant(QM31::from(M31::from(zeta)));
    let offset = (max_coeff / Q + 1) * Q;
    let offset_const = ctx.constant(QM31::from(M31::from(offset)));

    // Batch-reduce sums = a + b
    let sums: [Var; 4] = std::array::from_fn(|i| {
        circuits::eval!(ctx, (a[i].0) + (b[i].0))
    });
    let sum_bits = bits_for(2 * max_coeff);
    let new_a = mod_reduce_lazy_batch4(ctx, sums, sum_bits);

    // Batch-reduce diffs = b + offset - a
    let diffs: [Var; 4] = std::array::from_fn(|i| {
        circuits::eval!(ctx, ((b[i].0) + (offset_const)) - (a[i].0))
    });
    let sub_bits = bits_for(max_coeff + offset);
    let diff_reduced = mod_reduce_lazy_batch4(ctx, diffs, sub_bits);

    // Batch zeta * diff (each diff in [0, Q), product < Q^2 < 2^24)
    let mul_prods: [Var; 4] =
        std::array::from_fn(|i| pointwise_mul(ctx, diff_reduced[i].0, zeta_var));
    let new_b = mod_reduce_lazy_batch4(ctx, mul_prods, 24);

    (new_a, new_b)
}

/// Inverse NTT (Gentleman-Sande, decimation-in-frequency).
///
/// `input_max` is the maximum value of any coefficient in `poly`. After the optimized
/// forward NTT (which skips add-reduce), this is typically 8Q.
///
/// Optimized with batch range checks. The add output MUST be reduced because both
/// inputs can be unreduced (exponential growth otherwise).
pub fn inv_ntt<V: IValue + ZqWitness>(ctx: &mut Context<V>, poly: &mut RqPoly) {
    inv_ntt_ex(ctx, poly, 8 * Q)
}

/// Inverse NTT with explicit input bound.
pub fn inv_ntt_ex<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    poly: &mut RqPoly,
    input_max: u32,
) {
    let mut k = 127usize;
    let mut len = 2;

    // First layer uses input_max; after reduction, max drops to Q for subsequent layers.
    let mut max_coeff = input_max;

    while len <= 128 {
        let mut start = 0;
        while start < N {
            let zeta = ZETAS[k];
            k = k.wrapping_sub(1);

            if len >= 4 {
                let mut j = start;
                while j + 3 < start + len {
                    let a4: [ZqVar; 4] =
                        std::array::from_fn(|i| poly.coeffs[j + i]);
                    let b4: [ZqVar; 4] =
                        std::array::from_fn(|i| poly.coeffs[j + i + len]);

                    let (new_a4, new_b4) =
                        inv_butterfly_batch4(ctx, a4, b4, zeta, max_coeff);

                    for i in 0..4 {
                        poly.coeffs[j + i] = new_a4[i];
                        poly.coeffs[j + i + len] = new_b4[i];
                    }
                    j += 4;
                }
                while j < start + len {
                    let (a, b) = inv_butterfly_lazy(
                        ctx, poly.coeffs[j], poly.coeffs[j + len], zeta, max_coeff,
                    );
                    poly.coeffs[j] = a;
                    poly.coeffs[j + len] = b;
                    j += 1;
                }
            } else {
                for j in start..(start + len) {
                    let (a, b) = inv_butterfly_lazy(
                        ctx, poly.coeffs[j], poly.coeffs[j + len], zeta, max_coeff,
                    );
                    poly.coeffs[j] = a;
                    poly.coeffs[j + len] = b;
                }
            }
            start += 2 * len;
        }
        // After each layer, both outputs are reduced to [0, Q), so max resets.
        max_coeff = Q;
        len <<= 1;
    }

    // Scale all coefficients by 128^{-1} mod Q.
    // Coefficients are in [0, Q) after reduction. Product < Q * NTT_SCALE_INV < 2^24.
    let scale_input_bits = 24u32;
    let mut i = 0;
    while i + 3 < N {
        let c = ctx.constant(QM31::from(M31::from(NTT_SCALE_INV)));
        let prods: [Var; 4] = std::array::from_fn(|j| {
            pointwise_mul(ctx, poly.coeffs[i + j].0, c)
        });
        let results = mod_reduce_lazy_batch4(ctx, prods, scale_input_bits);
        for j in 0..4 {
            poly.coeffs[i + j] = results[j];
        }
        i += 4;
    }
    while i < N {
        poly.coeffs[i] = zq_mul_const(ctx, poly.coeffs[i], NTT_SCALE_INV);
        i += 1;
    }
}

/// Pointwise multiplication in NTT domain.
///
/// After optimized NTT, inputs may be up to 8Q (unreduced adds across 7 layers).
/// Products up to 64Q^2 ≈ 709M < P, so they fit in M31. Use input_bits=30.
pub fn ntt_pointwise_mul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &RqPoly,
    b: &RqPoly,
    a_max: u32,
    b_max: u32,
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
            a_max,
            b_max,
        );
        coeffs[4 * i] = r0;
        coeffs[4 * i + 1] = r1;

        let neg_zeta = Q - ZETAS[64 + i];
        let (r2, r3) = basemul(
            ctx,
            coeffs[4 * i + 2],
            coeffs[4 * i + 3],
            b.coeffs[4 * i + 2],
            b.coeffs[4 * i + 3],
            neg_zeta,
            a_max,
            b_max,
        );
        coeffs[4 * i + 2] = r2;
        coeffs[4 * i + 3] = r3;
    }
    RqPoly { coeffs }
}

/// Base multiplication with configurable input bounds.
fn basemul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a0: ZqVar,
    a1: ZqVar,
    b0: ZqVar,
    b1: ZqVar,
    gamma: u32,
    a_max: u32,
    b_max: u32,
) -> (ZqVar, ZqVar) {
    use crate::zq::zq_mul_ex;

    let mul_bits = bits_for(a_max * b_max);
    let a0_b0 = zq_mul_ex(ctx, a0, b0, mul_bits);
    let a1_b1 = zq_mul_ex(ctx, a1, b1, mul_bits);
    let a0_b1 = zq_mul_ex(ctx, a0, b1, mul_bits);
    let a1_b0 = zq_mul_ex(ctx, a1, b0, mul_bits);

    let a1_b1_gamma = zq_mul_const(ctx, a1_b1, gamma);
    let r0 = zq_add(ctx, a0_b0, a1_b1_gamma);
    let r1 = zq_add(ctx, a0_b1, a1_b0);
    (r0, r1)
}

/// Convenience: compute polynomial multiplication via NTT.
pub fn poly_mul<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    a: &mut RqPoly,
    b: &mut RqPoly,
) -> RqPoly {
    ntt(ctx, a);
    ntt(ctx, b);
    // After NTT, max coefficient is Q + 7*Q = 8Q (7 layers of unreduced adds)
    let ntt_max = 8 * Q;
    let mut result = ntt_pointwise_mul(ctx, a, b, ntt_max, ntt_max);
    inv_ntt(ctx, &mut result);
    result
}
