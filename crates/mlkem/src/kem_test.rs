use circuits::context::Context;
use circuits::ops::guess;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::constants::N;
use crate::kem::*;
use crate::ntt::{RqPoly, ntt};
use crate::zq::{zq_witness, ZqVar};

fn make_zero_poly(ctx: &mut Context<QM31>) -> RqPoly {
    let coeffs: [ZqVar; 256] = std::array::from_fn(|_| zq_witness(ctx, 0));
    RqPoly { coeffs }
}

fn make_zero_bytes(ctx: &mut Context<QM31>, n: usize) -> Vec<circuits::context::Var> {
    (0..n)
        .map(|_| guess(ctx, QM31::from(M31::from(0u32))))
        .collect()
}

/// Zero-noise encrypt-decrypt test (lower-level API, no hashing).
#[test]
fn test_encrypt_decrypt_zero_noise() {
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    let a_hat: Vec<Vec<RqPoly>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let mut p = if i == j {
                        let coeffs: [ZqVar; 256] = std::array::from_fn(|idx| {
                            zq_witness(&mut ctx, if idx == 0 { 1 } else { 0 })
                        });
                        RqPoly { coeffs }
                    } else {
                        make_zero_poly(&mut ctx)
                    };
                    ntt(&mut ctx, &mut p);
                    p
                })
                .collect()
        })
        .collect();

    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    let cbd_eta1 = (N as u32 * params.eta1 / 4) as usize;
    let cbd_eta2 = (N as u32 * params.eta2 / 4) as usize;
    let r_bytes: Vec<Vec<_>> = (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_eta1)).collect();
    let e1_bytes: Vec<Vec<_>> =
        (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_eta2)).collect();
    let e2_bytes = make_zero_bytes(&mut ctx, cbd_eta2);

    let (u_vec, v) = encrypt(
        &mut ctx, params, &a_hat, &t_hat, &message_bits, &r_bytes, &e1_bytes, &e2_bytes,
    );
    let recovered_bits = decrypt(&mut ctx, &s_hat, &u_vec, &v);

    let mut matching = 0;
    for i in 0..N {
        if ctx.get(message_bits[i]).0 .0 .0 == ctx.get(recovered_bits[i]).0 .0 .0 {
            matching += 1;
        }
    }
    assert_eq!(matching, N, "Expected perfect recovery with zero noise");

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
    eprintln!("Encrypt+Decrypt stats: {:?}", ctx.stats);
}

/// Full encaps+decaps test WITH Blake2s hashing.
/// This exercises the complete ML-KEM flow including hash-derived matrix A,
/// hash-derived noise, and hash-derived shared secret.
#[test]
fn test_encaps_decaps_with_hashing() {
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    // Public key seed ρ (2 QM31 = 32 bytes)
    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    // Secret key s (small, in NTT domain)
    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    // Public key t_hat (for zero secret, t = A*s + e = e ≈ 0)
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    // Message: 256 individual bit Vars
    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    // Encapsulate
    let (u_vec, v, ss_enc) = encaps(&mut ctx, params, &rho, &t_hat, &message_bits);

    // Decapsulate
    let (_recovered_msg, ss_dec) = decaps(&mut ctx, &s_hat, &u_vec, &v, &rho);

    // Verify shared secrets match
    assert_eq!(
        ctx.get(ss_enc.0),
        ctx.get(ss_dec.0),
        "Shared secret mismatch (lo)"
    );
    assert_eq!(
        ctx.get(ss_enc.1),
        ctx.get(ss_dec.1),
        "Shared secret mismatch (hi)"
    );

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();

    eprintln!("Encaps+Decaps with hashing stats: {:?}", ctx.stats);
}
