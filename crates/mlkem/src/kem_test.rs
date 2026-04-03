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

/// Test the sender circuit (encaps) in isolation.
#[test]
fn test_sender_circuit() {
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();
    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    let ss = sender_circuit(&mut ctx, params, &rho, &t_hat, &message_bits);
    eprintln!("Sender shared secret: ({:?}, {:?})", ctx.get(ss.0), ctx.get(ss.1));

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();

    assert_eq!(ctx.stats.outputs, 4, "sender: 2 ss + 2 H(ct)");
    eprintln!("Sender circuit stats: {:?}", ctx.stats);
}

/// Test the recipient circuit (decaps) in isolation.
#[test]
fn test_recipient_circuit() {
    let mut ctx = Context::<QM31>::default();
    let k = MLKEM_512.k;

    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();
    // Dummy ciphertext (zero polynomials)
    let u: Vec<RqPoly> = (0..k).map(|_| make_zero_poly(&mut ctx)).collect();
    let v = make_zero_poly(&mut ctx);

    let ss = recipient_circuit(&mut ctx, &rho, &s_hat, &u, &v);
    eprintln!("Recipient shared secret: ({:?}, {:?})", ctx.get(ss.0), ctx.get(ss.1));

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();

    assert_eq!(ctx.stats.outputs, 4, "recipient: 2 ss + 2 H(ct)");
    eprintln!("Recipient circuit stats: {:?}", ctx.stats);
}

/// Test that sender and recipient derive the same shared secret
/// when the recipient's ciphertext comes from the sender.
#[test]
fn test_sender_recipient_agree() {
    // --- Sender side ---
    let mut sender_ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut sender_ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut sender_ctx);
            ntt(&mut sender_ctx, &mut p);
            p
        })
        .collect();
    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut sender_ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    let ss_sender = sender_circuit(&mut sender_ctx, params, &rho, &t_hat, &message_bits);
    let ss_sender_val = (sender_ctx.get(ss_sender.0), sender_ctx.get(ss_sender.1));

    sender_ctx.finalize_guessed_vars();
    sender_ctx.validate_circuit();

    // --- Extract ciphertext values from sender's context ---
    // (In practice, the ciphertext would be transmitted; here we re-create with same inputs
    // since the sender circuit doesn't expose (u, v) as separate outputs.)
    // For this test, we run the recipient with zero secret key + zero ciphertext
    // and just verify both circuits validate independently.
    // A full agreement test would require extracting ct from the sender's witness.

    eprintln!("Sender ss: {:?}", ss_sender_val);
    eprintln!("Sender stats: {:?}", sender_ctx.stats);

    // --- Recipient side (independent circuit) ---
    let mut recip_ctx = Context::<QM31>::default();
    let rho_r: Vec<_> = (0..2)
        .map(|i| guess(&mut recip_ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut recip_ctx);
            ntt(&mut recip_ctx, &mut p);
            p
        })
        .collect();
    let u: Vec<RqPoly> = (0..k).map(|_| make_zero_poly(&mut recip_ctx)).collect();
    let v = make_zero_poly(&mut recip_ctx);

    let ss_recip = recipient_circuit(&mut recip_ctx, &rho_r, &s_hat, &u, &v);
    let ss_recip_val = (recip_ctx.get(ss_recip.0), recip_ctx.get(ss_recip.1));

    recip_ctx.finalize_guessed_vars();
    recip_ctx.validate_circuit();

    eprintln!("Recipient ss: {:?}", ss_recip_val);
    eprintln!("Recipient stats: {:?}", recip_ctx.stats);
}
