use circuits::blake::HashValue;
use circuits::context::{Context, Var};
use circuits::extract_bits::extract_bits;
use circuits::ivalue::IValue;
use circuits::ops::{eq, guess, pointwise_mul};
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

/// ML-KEM parameters for a given security level.
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

/// Derive the k×k matrix A in NTT domain from seed ρ using XOF (Blake2s).
///
/// Each A[i][j] is derived from XOF(ρ || j || i). Hash output M31 words are
/// reduced mod Q to produce NTT-domain coefficients.
pub fn derive_matrix_a<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    rho: &[Var],
    params: &MlKemParams,
) -> Vec<Vec<RqPoly>> {
    let k = params.k;
    // Need 256 coefficients per poly. Each XOF block gives 8 M31 words.
    // 256 / 8 = 32 blocks needed.
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

/// Derive noise polynomials from seed σ using PRF (Blake2s) + CBD.
pub fn derive_noise<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    sigma: &[Var],
    nonce_start: u8,
    count: usize,
    eta: u32,
) -> Vec<RqPoly> {
    let bytes_per_poly = (N as u32 * eta / 4) as usize;
    // Each PRF call produces 8 M31 words. We extract low byte from each.
    let words_per_poly = bytes_per_poly; // 1 byte per word
    let calls_per_poly = words_per_poly.div_ceil(8);

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

/// Reduce an M31 word to [0, 256) for byte extraction.
fn mod_reduce_byte<V: IValue + ZqWitness>(ctx: &mut Context<V>, x: Var) -> Var {
    let x_int = ctx.get(x).raw_u32();
    let r_int = x_int % 256;
    let q_int = x_int / 256;

    let r_var = guess(ctx, V::from_u32(r_int));
    let q_var = guess(ctx, V::from_u32(q_int));

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

/// Unpack a HashValue (2 QM31 = 8 M31 words) into individual Var M31 words.
fn unpack_hash_to_vars<V: IValue>(ctx: &mut Context<V>, hash: &HashValue<Var>) -> Vec<Var> {
    let s0 = Simd::from_packed(vec![hash.0], 4);
    let s1 = Simd::from_packed(vec![hash.1], 4);
    let mut words = Vec::with_capacity(8);
    for i in 0..4 { words.push(Simd::unpack_idx(ctx, &s0, i)); }
    for i in 0..4 { words.push(Simd::unpack_idx(ctx, &s1, i)); }
    words
}

/// ML-KEM Encapsulation circuit with Blake2s hashing.
///
/// `message_bits`: 256 individual bit Vars (same format as decrypt output).
/// Derives matrix A from ρ, noise from hash of message, encrypts,
/// and produces a shared secret.
pub fn encaps<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    params: &MlKemParams,
    rho: &[Var],
    t_hat: &[RqPoly],
    message_bits: &[Var],
) -> (Vec<RqPoly>, RqPoly, HashValue<Var>) {
    assert_eq!(message_bits.len(), N);
    let k = params.k;

    // H(pk)
    let h_pk = hash_h(ctx, rho);

    // G(m || H(pk)) — hash the message bits alongside pk hash
    let mut g_input = message_bits.to_vec();
    g_input.push(h_pk.0);
    g_input.push(h_pk.1);
    let (k_hash, r_hash) = hash_g(ctx, &g_input);

    // Derive matrix A from ρ
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

    // v = INTT(t_hat^T · NTT(r)) + e2 + decompress_1(m)
    let mut v_hat = inner_product_ntt(ctx, t_hat, &r_polys);
    inv_ntt(ctx, &mut v_hat);
    let msg_poly = message_to_poly(ctx, message_bits);
    let v_plus_e2 = poly_add(ctx, &v_hat, &e2_polys[0]);
    let v = poly_add(ctx, &v_plus_e2, &msg_poly);

    // Shared secret K' = H(K || H(ct))
    let ct_hash = hash_h(ctx, &[u_vec[0].coeffs[0].0, v.coeffs[0].0]);
    let shared_secret = hash_h(ctx, &[k_hash.0, k_hash.1, ct_hash.0, ct_hash.1]);

    (u_vec, v, shared_secret)
}

/// ML-KEM Decapsulation circuit.
pub fn decaps<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    s_hat: &[RqPoly],
    u: &[RqPoly],
    v: &RqPoly,
    rho: &[Var],
) -> (Vec<Var>, HashValue<Var>) {
    // Decrypt
    let message_bits = decrypt(ctx, s_hat, u, v);

    // Re-derive shared secret using same path as encaps
    let h_pk = hash_h(ctx, rho);
    let mut g_input: Vec<Var> = message_bits.clone();
    g_input.push(h_pk.0);
    g_input.push(h_pk.1);
    let (k_hash, _r_hash) = hash_g(ctx, &g_input);

    let ct_hash = hash_h(ctx, &[u[0].coeffs[0].0, v.coeffs[0].0]);
    let shared_secret = hash_h(ctx, &[k_hash.0, k_hash.1, ct_hash.0, ct_hash.1]);

    (message_bits, shared_secret)
}

/// Lower-level encrypt (takes pre-derived noise as byte Vars).
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
    let mut r_vec: Vec<RqPoly> = (0..k).map(|i| cbd(ctx, &r_bytes[i], params.eta1)).collect();
    let e1_vec: Vec<RqPoly> = (0..k).map(|i| cbd(ctx, &e1_bytes[i], params.eta2)).collect();
    let e2 = cbd(ctx, e2_bytes, params.eta2);

    for r in &mut r_vec { ntt(ctx, r); }

    let a_t: Vec<Vec<RqPoly>> = (0..k)
        .map(|i| (0..k).map(|j| a_hat[j][i].clone()).collect())
        .collect();
    let mut u_hat = matrix_vec_mul_ntt(ctx, &a_t, &r_vec);
    for u in &mut u_hat { inv_ntt(ctx, u); }
    let u_vec: Vec<RqPoly> = (0..k).map(|i| poly_add(ctx, &u_hat[i], &e1_vec[i])).collect();

    let mut v_hat = inner_product_ntt(ctx, t_hat, &r_vec);
    inv_ntt(ctx, &mut v_hat);
    let msg_poly = message_to_poly(ctx, message_bits);
    let v_tmp = poly_add(ctx, &v_hat, &e2);
    let v = poly_add(ctx, &v_tmp, &msg_poly);

    (u_vec, v)
}

fn message_to_poly<V: IValue + ZqWitness>(ctx: &mut Context<V>, bits: &[Var]) -> RqPoly {
    assert_eq!(bits.len(), N);
    let scale = ctx.constant(QM31::from(M31::from(1665u32)));
    let coeffs = std::array::from_fn(|i| ZqVar(pointwise_mul(ctx, bits[i], scale)));
    RqPoly { coeffs }
}

pub fn decrypt<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    s_hat: &[RqPoly],
    u: &[RqPoly],
    v: &RqPoly,
) -> Vec<Var> {
    let mut u_ntt: Vec<RqPoly> = u.to_vec();
    for u_poly in &mut u_ntt { ntt(ctx, u_poly); }

    let mut inner = inner_product_ntt(ctx, s_hat, &u_ntt);
    inv_ntt(ctx, &mut inner);
    let w = poly_sub(ctx, v, &inner);

    (0..N)
        .map(|i| {
            let reduced = mod_reduce_lazy(ctx, w.coeffs[i].0, 14);
            compress(ctx, reduced, 1)
        })
        .collect()
}
