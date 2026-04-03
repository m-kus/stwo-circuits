use circuits::context::Context;
use circuits::ops::guess;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::constants::{N, Q};
use crate::sampling::cbd;

#[test]
fn test_cbd_eta2_zeros() {
    let mut ctx = Context::<QM31>::default();

    // All-zero input: each coefficient should be 0 (0 - 0)
    let bytes: Vec<_> = (0..128)
        .map(|_| guess(&mut ctx, QM31::from(M31::from(0u32))))
        .collect();

    let poly = cbd(&mut ctx, &bytes, 2);

    for i in 0..N {
        assert_eq!(ctx.get(poly.coeffs[i].0).0 .0 .0, 0, "expected 0 at index {i}");
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_cbd_eta2_all_ones() {
    let mut ctx = Context::<QM31>::default();

    // All 0xFF bytes: every bit is 1
    // For η=2: each coeff = (1+1) - (1+1) = 0
    let bytes: Vec<_> = (0..128)
        .map(|_| guess(&mut ctx, QM31::from(M31::from(0xFFu32))))
        .collect();

    let poly = cbd(&mut ctx, &bytes, 2);

    for i in 0..N {
        assert_eq!(ctx.get(poly.coeffs[i].0).0 .0 .0, 0, "expected 0 at index {i}");
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_cbd_eta2_known_pattern() {
    let mut ctx = Context::<QM31>::default();

    // First byte = 0b00000011 = 3 (bits: 1,1,0,0,0,0,0,0 in LSB order)
    // Coefficient 0: bits 0-1 for a (1+1=2), bits 2-3 for b (0+0=0) → 2 - 0 = 2
    // Coefficient 1: bits 4-5 for a (0+0=0), bits 6-7 for b (0+0=0) → 0 - 0 = 0
    let mut byte_vals = vec![0u32; 128];
    byte_vals[0] = 3; // 0b00000011

    let bytes: Vec<_> = byte_vals
        .iter()
        .map(|&v| guess(&mut ctx, QM31::from(M31::from(v))))
        .collect();

    let poly = cbd(&mut ctx, &bytes, 2);

    let coeff0 = ctx.get(poly.coeffs[0].0).0 .0 .0;
    assert_eq!(coeff0, 2, "expected coeff[0] = 2, got {coeff0}");

    let coeff1 = ctx.get(poly.coeffs[1].0).0 .0 .0;
    assert_eq!(coeff1, 0, "expected coeff[1] = 0, got {coeff1}");

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_cbd_eta2_negative() {
    let mut ctx = Context::<QM31>::default();

    // Byte = 0b00001100 = 12 (bits: 0,0,1,1,0,0,0,0 in LSB order)
    // Coefficient 0: bits 0-1 for a (0+0=0), bits 2-3 for b (1+1=2) → 0 - 2 = -2 mod Q = 3327
    let mut byte_vals = vec![0u32; 128];
    byte_vals[0] = 12;

    let bytes: Vec<_> = byte_vals
        .iter()
        .map(|&v| guess(&mut ctx, QM31::from(M31::from(v))))
        .collect();

    let poly = cbd(&mut ctx, &bytes, 2);

    let coeff0 = ctx.get(poly.coeffs[0].0).0 .0 .0;
    assert_eq!(coeff0 % Q, Q - 2, "expected coeff[0] = Q-2 = {}, got {coeff0}", Q - 2);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}
