use circuits::blake::HashValue;
use circuits::context::{Context, Var};
use circuits::extract_bits::extract_bits;
use circuits::ivalue::IValue;
use circuits::ops::{eq, guess, output, pointwise_mul};
use circuits::simd::Simd;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::compress::compress;
use crate::constants::N;
use crate::hash::{hash_g, hash_h, prf, xof};
use crate::ntt::{RqPoly, inv_ntt, ntt};
use crate::poly::{inner_product_ntt, matrix_vec_mul_ntt, poly_add, poly_sub};
use crate::sampling::cbd;
use crate::zq::{ZqVar, ZqWitness, mod_reduce_lazy};

#[cfg(test)]
#[path = "kem_test.rs"]
pub mod test;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

pub struct MlKemParams {
    pub k: usize,
    pub eta1: u32,
    pub eta2: u32,
    pub du: u32,
    pub dv: u32,
}

pub const MLKEM_512: MlKemParams = MlKemParams { k: 2, eta1: 3, eta2: 2, du: 10, dv: 4 };
pub const MLKEM_768: MlKemParams = MlKemParams { k: 3, eta1: 2, eta2: 2, du: 10, dv: 4 };
pub const MLKEM_1024: MlKemParams = MlKemParams { k: 4, eta1: 2, eta2: 2, du: 11, dv: 5 };

// ---------------------------------------------------------------------------
// Circuit 1: Sender — derive shared key when sending data
// ---------------------------------------------------------------------------

/// Sender circuit (Encaps).
///
/// The sender fetches the recipient's public key `(ρ, t_hat)` from a PKI,
/// picks a random message m, and produces a ciphertext + shared secret.
///
/// **Public inputs** (from PKI / on-chain):
///   - ρ          — recipient's public key seed (Var, 2 QM31 = 32 bytes)
///   - t_hat      — recipient's public key vector in NTT domain (k polynomials)
///
/// **Witness** (private to sender):
///   - message_bits — 256-bit random message
///   - (all derived noise, NTT intermediates, etc.)
///
/// **Public outputs**:
///   - shared_secret (2 QM31)
///   - ciphertext hash H(ct) (2 QM31)
pub fn sender_circuit<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    params: &MlKemParams,
    rho: &[Var],
    t_hat: &[RqPoly],
    message_bits: &[Var],
) -> HashValue<Var> {
    assert_eq!(message_bits.len(), N);
    let k = params.k;

    // H(pk)
    let h_pk = hash_h(ctx, rho);

    // G(m || H(pk)) → (K, r)
    let mut g_input = message_bits.to_vec();
    g_input.push(h_pk.0);
    g_input.push(h_pk.1);
    let (k_hash, r_hash) = hash_g(ctx, &g_input);

    // Derive A from ρ
    let a_hat = derive_matrix_a(ctx, rho, params);

    // Derive noise from r
    let sigma = vec![r_hash.0, r_hash.1];
    let mut r_polys = derive_noise(ctx, &sigma, 0, k, params.eta1);
    let e1_polys = derive_noise(ctx, &sigma, k as u8, k, params.eta2);
    let e2_polys = derive_noise(ctx, &sigma, (2 * k) as u8, 1, params.eta2);

    // NTT(r)
    for r in &mut r_polys {
        ntt(ctx, r);
    }

    // u = INTT(A^T · NTT(r)) + e1
    let a_t: Vec<Vec<RqPoly>> = (0..k)
        .map(|i| (0..k).map(|j| a_hat[j][i].clone()).collect())
        .collect();
    let mut u_hat = matrix_vec_mul_ntt(ctx, &a_t, &r_polys);
    for u in &mut u_hat {
        inv_ntt(ctx, u);
    }
    let u_vec: Vec<RqPoly> = (0..k)
        .map(|i| poly_add(ctx, &u_hat[i], &e1_polys[i]))
        .collect();

    // v = INTT(t^T · NTT(r)) + e2 + Decompress_1(m)
    let mut v_hat = inner_product_ntt(ctx, t_hat, &r_polys);
    inv_ntt(ctx, &mut v_hat);
    let msg_poly = message_to_poly(ctx, message_bits);
    let v_plus_e2 = poly_add(ctx, &v_hat, &e2_polys[0]);
    let v = poly_add(ctx, &v_plus_e2, &msg_poly);

    // Shared secret = H(K || H(ct))
    let ct_hash = hash_h(ctx, &[u_vec[0].coeffs[0].0, v.coeffs[0].0]);
    let shared_secret = hash_h(ctx, &[k_hash.0, k_hash.1, ct_hash.0, ct_hash.1]);

    // Public outputs
    output(ctx, shared_secret.0);
    output(ctx, shared_secret.1);
    output(ctx, ct_hash.0);
    output(ctx, ct_hash.1);

    shared_secret
}

// ---------------------------------------------------------------------------
// Circuit 2: Recipient — derive shared key when receiving data
// ---------------------------------------------------------------------------

/// Recipient circuit (Decaps).
///
/// The recipient receives a ciphertext (u, v) from the sender and uses their
/// secret key to derive the same shared secret.
///
/// **Public inputs**:
///   - ρ       — own public key seed
///   - t_hat   — own public key vector (NTT domain)
///   - u       — ciphertext vector (k polynomials, coefficient domain)
///   - v       — ciphertext polynomial (coefficient domain)
///
/// **Witness** (private to recipient):
///   - s_hat   — secret key (NTT domain)
///
/// **Public outputs**:
///   - shared_secret (2 QM31)
///   - ciphertext hash H(ct) (2 QM31)
pub fn recipient_circuit<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    rho: &[Var],
    s_hat: &[RqPoly],
    u: &[RqPoly],
    v: &RqPoly,
) -> HashValue<Var> {
    // Decrypt: m' = Compress_1(v - INTT(s^T · NTT(u)))
    let mut u_ntt: Vec<RqPoly> = u.to_vec();
    for u_poly in &mut u_ntt {
        ntt(ctx, u_poly);
    }
    let mut inner = inner_product_ntt(ctx, s_hat, &u_ntt);
    inv_ntt(ctx, &mut inner);
    let w = poly_sub(ctx, v, &inner);
    let message_bits: Vec<Var> = (0..N)
        .map(|i| {
            let reduced = mod_reduce_lazy(ctx, w.coeffs[i].0, 14);
            compress(ctx, reduced, 1)
        })
        .collect();

    // Re-derive shared secret: G(m' || H(pk)) → (K, _)
    let h_pk = hash_h(ctx, rho);
    let mut g_input: Vec<Var> = message_bits;
    g_input.push(h_pk.0);
    g_input.push(h_pk.1);
    let (k_hash, _r_hash) = hash_g(ctx, &g_input);

    // Shared secret = H(K || H(ct))
    let ct_hash = hash_h(ctx, &[u[0].coeffs[0].0, v.coeffs[0].0]);
    let shared_secret = hash_h(ctx, &[k_hash.0, k_hash.1, ct_hash.0, ct_hash.1]);

    // Public outputs
    output(ctx, shared_secret.0);
    output(ctx, shared_secret.1);
    output(ctx, ct_hash.0);
    output(ctx, ct_hash.1);

    shared_secret
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn derive_matrix_a<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    rho: &[Var],
    params: &MlKemParams,
) -> Vec<Vec<RqPoly>> {
    let k = params.k;
    let blocks_per_poly = 32;

    (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let j_var = ctx.constant(QM31::from(M31::from(j as u32)));
                    let i_var = ctx.constant(QM31::from(M31::from(i as u32)));
                    let mut seed = rho.to_vec();
                    seed.push(j_var);
                    seed.push(i_var);
                    let blocks = xof(ctx, &seed, blocks_per_poly);

                    let mut coeffs = [ZqVar(ctx.zero()); N];
                    let mut idx = 0;
                    for block in &blocks {
                        if idx >= N { break; }
                        let words = unpack_hash_to_vars(ctx, block);
                        for word in &words {
                            if idx >= N { break; }
                            coeffs[idx] = mod_reduce_lazy(ctx, *word, 31);
                            idx += 1;
                        }
                    }
                    RqPoly { coeffs }
                })
                .collect()
        })
        .collect()
}

fn derive_noise<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    sigma: &[Var],
    nonce_start: u8,
    count: usize,
    eta: u32,
) -> Vec<RqPoly> {
    let bytes_per_poly = (N as u32 * eta / 4) as usize;
    let calls_per_poly = bytes_per_poly.div_ceil(8);

    (0..count)
        .map(|i| {
            let nonce = nonce_start + i as u8;
            let mut byte_vars = Vec::with_capacity(bytes_per_poly);
            for call_idx in 0..calls_per_poly {
                let mut key = sigma.to_vec();
                key.push(ctx.constant(QM31::from(M31::from(nonce as u32))));
                key.push(ctx.constant(QM31::from(M31::from(call_idx as u32))));
                let hash = prf(ctx, &key, 0);
                let words = unpack_hash_to_vars(ctx, &hash);
                for word in &words {
                    if byte_vars.len() >= bytes_per_poly { break; }
                    byte_vars.push(mod_reduce_byte(ctx, *word));
                }
            }
            byte_vars.truncate(bytes_per_poly);
            cbd(ctx, &byte_vars, eta)
        })
        .collect()
}

fn mod_reduce_byte<V: IValue + ZqWitness>(ctx: &mut Context<V>, x: Var) -> Var {
    let x_int = ctx.get(x).raw_u32();
    let r_var = guess(ctx, V::from_u32(x_int % 256));
    let q_var = guess(ctx, V::from_u32(x_int / 256));

    let c256 = ctx.constant(QM31::from(M31::from(256u32)));
    let q256 = pointwise_mul(ctx, q_var, c256);
    let reconstructed = circuits::eval!(ctx, (q256) + (r_var));
    eq(ctx, reconstructed, x);

    let r_simd = Simd::from_packed(vec![r_var], 1);
    let _bits = extract_bits(ctx, &r_simd, 8);
    let q_simd = Simd::from_packed(vec![q_var], 1);
    let _bits = extract_bits(ctx, &q_simd, 23);
    r_var
}

fn unpack_hash_to_vars<V: IValue>(ctx: &mut Context<V>, hash: &HashValue<Var>) -> Vec<Var> {
    let s0 = Simd::from_packed(vec![hash.0], 4);
    let s1 = Simd::from_packed(vec![hash.1], 4);
    let mut words = Vec::with_capacity(8);
    for i in 0..4 { words.push(Simd::unpack_idx(ctx, &s0, i)); }
    for i in 0..4 { words.push(Simd::unpack_idx(ctx, &s1, i)); }
    words
}

fn message_to_poly<V: IValue + ZqWitness>(ctx: &mut Context<V>, bits: &[Var]) -> RqPoly {
    assert_eq!(bits.len(), N);
    let scale = ctx.constant(QM31::from(M31::from(1665u32)));
    let coeffs = std::array::from_fn(|i| ZqVar(pointwise_mul(ctx, bits[i], scale)));
    RqPoly { coeffs }
}
