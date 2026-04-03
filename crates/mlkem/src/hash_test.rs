use circuits::context::Context;
use circuits::ops::guess;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::hash::*;

#[test]
fn test_hash_h() {
    let mut ctx = Context::<QM31>::default();
    ctx.enable_assert_eq_on_eval();

    let input: Vec<_> = (0..4)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(i as u32 + 10))))
        .collect();

    let output = hash_h(&mut ctx, &input, 64);
    // Just verify the circuit is valid
    let _ = ctx.get(output.0);
    let _ = ctx.get(output.1);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_hash_g() {
    let mut ctx = Context::<QM31>::default();

    let input: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(i as u32 + 42))))
        .collect();

    let (h0, h1) = hash_g(&mut ctx, &input, 32);
    // The two halves should be different (different domain separation tags)
    assert_ne!(ctx.get(h0.0), ctx.get(h1.0));

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_prf() {
    let mut ctx = Context::<QM31>::default();

    let key: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(i as u32 + 100))))
        .collect();

    let out0 = prf(&mut ctx, &key, 32, 0);
    let out1 = prf(&mut ctx, &key, 32, 1);
    // Different nonces should give different outputs
    assert_ne!(ctx.get(out0.0), ctx.get(out1.0));

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_xof() {
    let mut ctx = Context::<QM31>::default();

    let seed: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(i as u32))))
        .collect();

    let blocks = xof(&mut ctx, &seed, 32, 3);
    assert_eq!(blocks.len(), 3);

    // Each block should be different (different counter)
    assert_ne!(ctx.get(blocks[0].0), ctx.get(blocks[1].0));
    assert_ne!(ctx.get(blocks[1].0), ctx.get(blocks[2].0));

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}
