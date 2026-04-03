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

/// Zero-noise encrypt-decrypt test.
///
/// With zero noise (e=0, e1=0, e2=0, r has specific structure),
/// decryption should be exact.
#[test]
fn test_encrypt_decrypt_zero_noise() {
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    // Matrix A in NTT domain (identity-like for simplicity)
    let a_hat: Vec<Vec<RqPoly>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let mut p = if i == j {
                        // Diagonal: use a simple polynomial
                        let coeffs: [ZqVar; 256] =
                            std::array::from_fn(|idx| zq_witness(&mut ctx, if idx == 0 { 1 } else { 0 }));
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

    // Secret key s (all zeros for simplicity)
    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    // t_hat = A * s + e = 0 (since s=0 and e=0)
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    // Message: alternating bits
    let message_bits: Vec<_> = (0..N)
        .map(|i| {
            let bit = (i % 2) as u32;
            guess(&mut ctx, QM31::from(M31::from(bit)))
        })
        .collect();

    // All-zero randomness → all noise polynomials will be zero
    let cbd_bytes_eta1 = (N as u32 * params.eta1 / 4) as usize;
    let cbd_bytes_eta2 = (N as u32 * params.eta2 / 4) as usize;

    let r_bytes: Vec<Vec<_>> = (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_bytes_eta1)).collect();
    let e1_bytes: Vec<Vec<_>> =
        (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_bytes_eta2)).collect();
    let e2_bytes = make_zero_bytes(&mut ctx, cbd_bytes_eta2);

    // Encrypt
    let (u_vec, v) = encrypt(
        &mut ctx, params, &a_hat, &t_hat, &message_bits, &r_bytes, &e1_bytes, &e2_bytes,
    );

    // Decrypt
    let recovered_bits = decrypt(&mut ctx, &s_hat, &u_vec, &v);

    // With zero noise and zero secret, v = 0 + 0 + decompress_1(m) = m * 1665
    // decrypt computes: w = v - INTT(s^T * NTT(u)) = v - 0 = v
    // compress_1(m * 1665) should recover m
    let mut matching = 0;
    for i in 0..N {
        let original = ctx.get(message_bits[i]).0 .0 .0;
        let recovered = ctx.get(recovered_bits[i]).0 .0 .0;
        if original == recovered {
            matching += 1;
        }
    }

    eprintln!("Zero-noise recovery: {matching}/{N} bits match");
    assert_eq!(matching, N, "Expected perfect recovery with zero noise");

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();

    eprintln!("KEM circuit stats: {:?}", ctx.stats);
}
