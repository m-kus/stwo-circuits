use circuits::context::Context;
use circuits::ops::guess;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::compress::*;
use crate::constants::Q;
use crate::zq::zq_witness;

#[test]
fn test_compress_d1() {
    let mut ctx = Context::<QM31>::default();

    // compress_1(0) = round(2/3329 * 0) = 0
    let x = zq_witness(&mut ctx, 0);
    let y = compress(&mut ctx, x, 1);
    assert_eq!(ctx.get(y).0 .0 .0, 0);

    // compress_1(1665) = round(2/3329 * 1665) ≈ round(1.0003) = 1
    let x = zq_witness(&mut ctx, 1665);
    let y = compress(&mut ctx, x, 1);
    assert_eq!(ctx.get(y).0 .0 .0, 1);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_compress_d4() {
    let mut ctx = Context::<QM31>::default();

    // compress_4(0) = 0
    let x = zq_witness(&mut ctx, 0);
    let y = compress(&mut ctx, x, 4);
    assert_eq!(ctx.get(y).0 .0 .0, 0);

    // compress_4(Q/2) ≈ round(16/3329 * 1664) ≈ round(7.998) = 8
    let x = zq_witness(&mut ctx, 1664);
    let y = compress(&mut ctx, x, 4);
    assert_eq!(ctx.get(y).0 .0 .0, 8);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_decompress_d1() {
    let mut ctx = Context::<QM31>::default();

    // decompress_1(0) = round(3329/2 * 0) = 0
    let y = guess(&mut ctx, QM31::from(M31::from(0u32)));
    let x = decompress(&mut ctx, y, 1);
    assert_eq!(ctx.get(x.0).0 .0 .0, 0);

    // decompress_1(1) = round(3329/2) = 1665
    let y = guess(&mut ctx, QM31::from(M31::from(1u32)));
    let x = decompress(&mut ctx, y, 1);
    assert_eq!(ctx.get(x.0).0 .0 .0, 1665);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_compress_decompress_roundtrip() {
    let mut ctx = Context::<QM31>::default();

    // Compress then decompress should approximate the original
    for val in [0, 100, 500, 1000, 1664, 2000, 3000, 3328] {
        let x = zq_witness(&mut ctx, val);
        let y = compress(&mut ctx, x, 10);
        let x_prime = decompress(&mut ctx, y, 10);

        let original = val;
        let recovered = ctx.get(x_prime.0).0 .0 .0;
        // With d=10, error should be at most ceil(Q / 2^11) ≈ 2
        let diff = if recovered > original {
            recovered - original
        } else {
            original - recovered
        };
        assert!(
            diff <= 2 || (Q - diff) <= 2,
            "roundtrip error too large for {val}: got {recovered}, diff {diff}"
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}
