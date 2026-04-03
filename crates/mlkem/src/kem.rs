use circuits::context::{Context, Var};
use circuits::ivalue::IValue;
use circuits::ops::pointwise_mul;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::compress::compress;
use crate::constants::N;
use crate::ntt::{RqPoly, inv_ntt, ntt};
use crate::poly::{inner_product_ntt, matrix_vec_mul_ntt, poly_add, poly_sub};
use crate::sampling::cbd;
use crate::zq::{ZqVar, ZqWitness, mod_reduce_lazy};

#[cfg(test)]
#[path = "kem_test.rs"]
pub mod test;

/// ML-KEM parameters for a given security level.
pub struct MlKemParams {
    /// Dimension of the module lattice.
    pub k: usize,
    /// CBD parameter for secret/noise.
    pub eta1: u32,
    /// CBD parameter for ciphertext noise.
    pub eta2: u32,
    /// Compression parameter for vector u.
    pub du: u32,
    /// Compression parameter for polynomial v.
    pub dv: u32,
}

/// ML-KEM-512 parameters.
pub const MLKEM_512: MlKemParams = MlKemParams { k: 2, eta1: 3, eta2: 2, du: 10, dv: 4 };

/// ML-KEM-768 parameters.
pub const MLKEM_768: MlKemParams = MlKemParams { k: 3, eta1: 2, eta2: 2, du: 10, dv: 4 };

/// ML-KEM-1024 parameters.
pub const MLKEM_1024: MlKemParams = MlKemParams { k: 4, eta1: 2, eta2: 2, du: 11, dv: 5 };

/// ML-KEM encryption (inner operation of Encaps).
///
/// Given:
/// - a_hat: k×k matrix A in NTT domain
/// - t_hat: public key vector t in NTT domain
/// - message_bits: 256-bit message as Var bits
/// - r_bytes, e1_bytes, e2_bytes: randomness for CBD sampling
///
/// Computes:
/// 1. Sample noise: r_vec, e1, e2 via CBD
/// 2. u = NTT^{-1}(A^T · NTT(r_vec)) + e1
/// 3. v = NTT^{-1}(t_hat^T · NTT(r_vec)) + e2 + decompress_1(m)
///
/// Returns (u, v) in coefficient domain.
pub fn encrypt<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    params: &MlKemParams,
    a_hat: &[Vec<RqPoly>],
    t_hat: &[RqPoly],
    message_bits: &[Var],
    r_bytes: &[Vec<Var>],
    e1_bytes: &[Vec<Var>],
    e2_bytes: &[Var],
) -> (Vec<RqPoly>, RqPoly) {
    let k = params.k;

    // Sample r vector via CBD(η₁)
    let mut r_vec: Vec<RqPoly> = (0..k)
        .map(|i| cbd(ctx, &r_bytes[i], params.eta1))
        .collect();

    // Sample e1 vector via CBD(η₂)
    let e1_vec: Vec<RqPoly> = (0..k)
        .map(|i| cbd(ctx, &e1_bytes[i], params.eta2))
        .collect();

    // Sample e2 via CBD(η₂)
    let e2 = cbd(ctx, e2_bytes, params.eta2);

    // NTT(r)
    for r in &mut r_vec {
        ntt(ctx, r);
    }

    // u = INTT(A^T · NTT(r)) + e1
    let a_t: Vec<Vec<RqPoly>> = (0..k)
        .map(|i| (0..k).map(|j| a_hat[j][i].clone()).collect())
        .collect();

    let mut u_hat = matrix_vec_mul_ntt(ctx, &a_t, &r_vec);
    for u in &mut u_hat {
        inv_ntt(ctx, u);
    }
    let u_vec: Vec<RqPoly> = (0..k)
        .map(|i| poly_add(ctx, &u_hat[i], &e1_vec[i]))
        .collect();

    // v = INTT(t_hat^T · NTT(r)) + e2 + decompress_1(m)
    let mut v_hat = inner_product_ntt(ctx, t_hat, &r_vec);
    inv_ntt(ctx, &mut v_hat);

    let msg_poly = message_to_poly(ctx, message_bits);
    let v_plus_e2 = poly_add(ctx, &v_hat, &e2);
    let v = poly_add(ctx, &v_plus_e2, &msg_poly);

    (u_vec, v)
}

/// Convert a 256-bit message (as Var bits) to a polynomial via decompress_1.
/// Each bit b maps to b * round(Q/2) = b * 1665.
fn message_to_poly<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    bits: &[Var],
) -> RqPoly {
    assert_eq!(bits.len(), N);
    let scale = ctx.constant(QM31::from(M31::from(1665u32)));
    let coeffs = std::array::from_fn(|i| {
        ZqVar(pointwise_mul(ctx, bits[i], scale))
    });
    RqPoly { coeffs }
}

/// ML-KEM decryption (inner operation of Decaps).
///
/// Given secret key s_hat (NTT domain), ciphertext (u, v):
/// 1. Compute w = v - INTT(s_hat^T · NTT(u))
/// 2. Compress_1(w) to recover message
///
/// Returns 256 message bits.
pub fn decrypt<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    s_hat: &[RqPoly],
    u: &[RqPoly],
    v: &RqPoly,
) -> Vec<Var> {
    // NTT(u)
    let mut u_ntt: Vec<RqPoly> = u.to_vec();
    for u_poly in &mut u_ntt {
        ntt(ctx, u_poly);
    }

    // w = v - INTT(s_hat^T · NTT(u))
    let mut inner = inner_product_ntt(ctx, s_hat, &u_ntt);
    inv_ntt(ctx, &mut inner);
    let w = poly_sub(ctx, v, &inner);

    // Recover message: compress_1(w_i) for each coefficient
    (0..N)
        .map(|i| {
            // Coefficients may be unreduced after poly_sub, reduce first
            let reduced = mod_reduce_lazy(ctx, w.coeffs[i].0, 14);
            compress(ctx, reduced, 1)
        })
        .collect()
}
