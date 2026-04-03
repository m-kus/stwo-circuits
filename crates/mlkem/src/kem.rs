use circuits::blake::HashValue;
use circuits::context::{Context, Var};
use circuits::extract_bits::extract_bits;
use circuits::ivalue::IValue;
use circuits::ops::{output, pointwise_mul};
use circuits::simd::Simd;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::compress::compress;
use crate::constants::N;
use crate::hash::{hash_g, hash_h, prf};
use crate::ntt::{RqPoly, inv_ntt, ntt};
use crate::poly::{inner_product_ntt, matrix_vec_mul_ntt, poly_add, poly_sub};
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
    a_hat: &[Vec<RqPoly>],  // pre-computed matrix A (verifiable from ρ outside the proof)
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

/// Derive noise polynomials by extracting bits directly from PRF hash outputs.
/// Eliminates the expensive mod_reduce_byte step.
fn derive_noise<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    sigma: &[Var],
    nonce_start: u8,
    count: usize,
    eta: u32,
) -> Vec<RqPoly> {
    let bits_per_poly = N * 2 * eta as usize; // CBD needs 2*eta bits per coefficient

    // Each M31 hash word provides 31 usable bits.
    // Process 4 words at a time via Simd(4) → 4×31 = 124 bits per batch.
    let batches_per_poly = bits_per_poly.div_ceil(124);
    let words_per_poly = batches_per_poly * 4;
    let prf_calls_per_poly = words_per_poly.div_ceil(8); // 8 words per PRF call

    (0..count)
        .map(|i| {
            let nonce = nonce_start + i as u8;
            let mut all_bits: Vec<Var> = Vec::with_capacity(bits_per_poly + 128);

            for call_idx in 0..prf_calls_per_poly {
                let mut key = sigma.to_vec();
                key.push(ctx.constant(QM31::from(M31::from(nonce as u32))));
                key.push(ctx.constant(QM31::from(M31::from(call_idx as u32))));
                let hash = prf(ctx, &key, 0);
                let words = unpack_hash_to_vars(ctx, &hash);

                // Extract 31 bits from each word. Process in groups of 4 via Simd(4).
                let mut w = 0;
                while w + 3 < words.len() && all_bits.len() < bits_per_poly {
                    let packed = Simd::from_packed(
                        vec![pack_4_m31_vars(ctx, &words[w..w + 4])],
                        4,
                    );
                    let bit_simds = extract_bits(ctx, &packed, 31);
                    // Unpack bits from Simd(4): each bit_simd has 4 lanes
                    for bit_simd in &bit_simds {
                        for lane in 0..4 {
                            if all_bits.len() >= bits_per_poly { break; }
                            all_bits.push(Simd::unpack_idx(ctx, bit_simd, lane));
                        }
                    }
                    w += 4;
                }
                // Handle remaining words individually
                while w < words.len() && all_bits.len() < bits_per_poly {
                    let simd = Simd::from_packed(vec![words[w]], 1);
                    let bit_simds = extract_bits(ctx, &simd, 31);
                    for bit_simd in &bit_simds {
                        if all_bits.len() >= bits_per_poly { break; }
                        all_bits.push(Simd::unpack(ctx, bit_simd)[0]);
                    }
                    w += 1;
                }
            }

            all_bits.truncate(bits_per_poly);
            cbd_from_bits(ctx, &all_bits, eta)
        })
        .collect()
}

/// Pack 4 individual M31-valued Vars into a single QM31 Var for Simd(4).
fn pack_4_m31_vars<V: IValue>(ctx: &mut Context<V>, vars: &[Var]) -> Var {
    assert!(vars.len() >= 4);
    let unit1 = ctx.constant(circuits::ivalue::qm31_from_u32s(0, 1, 0, 0));
    let unit2 = ctx.constant(circuits::ivalue::qm31_from_u32s(0, 0, 1, 0));
    let unit3 = ctx.constant(circuits::ivalue::qm31_from_u32s(0, 0, 0, 1));
    let t1 = circuits::eval!(ctx, (vars[1]) * (unit1));
    let t2 = circuits::eval!(ctx, (vars[2]) * (unit2));
    let t3 = circuits::eval!(ctx, (vars[3]) * (unit3));
    let s01 = circuits::eval!(ctx, (vars[0]) + (t1));
    let s23 = circuits::eval!(ctx, (t2) + (t3));
    circuits::eval!(ctx, (s01) + (s23))
}

/// CBD sampling from a flat bit vector (no byte abstraction).
fn cbd_from_bits<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    bits: &[Var],
    eta: u32,
) -> RqPoly {
    assert!(bits.len() >= N * 2 * eta as usize);
    let mut coeffs = [ZqVar(ctx.zero()); N];
    for i in 0..N {
        let offset = i * 2 * eta as usize;
        // sum_a = sum of first eta bits
        let mut sum_a = ctx.zero();
        for j in 0..eta as usize {
            sum_a = circuits::eval!(ctx, (sum_a) + (bits[offset + j]));
        }
        // sum_b = sum of next eta bits
        let mut sum_b = ctx.zero();
        for j in 0..eta as usize {
            sum_b = circuits::eval!(ctx, (sum_b) + (bits[offset + eta as usize + j]));
        }
        coeffs[i] = crate::zq::zq_sub(ctx, ZqVar(sum_a), ZqVar(sum_b));
    }
    RqPoly { coeffs }
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
