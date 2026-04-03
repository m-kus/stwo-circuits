use circuits::context::Context;
use stwo::core::fields::qm31::QM31;

use crate::constants::{N, Q};
use crate::ntt::{RqPoly, ntt};
use crate::poly::*;
use crate::zq::{zq_witness, ZqVar};

fn make_poly(ctx: &mut Context<QM31>, vals: &[u32; 256]) -> RqPoly {
    let coeffs: [ZqVar; 256] = std::array::from_fn(|i| zq_witness(ctx, vals[i]));
    RqPoly { coeffs }
}

fn extract_vals(ctx: &Context<QM31>, poly: &RqPoly) -> [u32; 256] {
    std::array::from_fn(|i| ctx.get(poly.coeffs[i].0).0 .0 .0)
}

#[test]
fn test_poly_add() {
    let mut ctx = Context::<QM31>::default();
    let a_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 7) % Q);
    let b_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 11 + 5) % Q);

    let a = make_poly(&mut ctx, &a_vals);
    let b = make_poly(&mut ctx, &b_vals);
    let c = poly_add(&mut ctx, &a, &b);
    let result = extract_vals(&ctx, &c);

    for i in 0..N {
        assert_eq!(result[i] % Q, (a_vals[i] + b_vals[i]) % Q, "poly_add failed at {i}");
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_poly_sub() {
    let mut ctx = Context::<QM31>::default();
    let a_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 7) % Q);
    let b_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 11 + 5) % Q);

    let a = make_poly(&mut ctx, &a_vals);
    let b = make_poly(&mut ctx, &b_vals);
    let c = poly_sub(&mut ctx, &a, &b);
    let result = extract_vals(&ctx, &c);

    for i in 0..N {
        assert_eq!(
            result[i] % Q,
            (a_vals[i] + Q - b_vals[i]) % Q,
            "poly_sub failed at {i}"
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_poly_mul() {
    let mut ctx = Context::<QM31>::default();
    let a_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 3 + 1) % Q);
    let b_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 5 + 2) % Q);

    let a = make_poly(&mut ctx, &a_vals);
    let b = make_poly(&mut ctx, &b_vals);
    let result = poly_mul(&mut ctx, &a, &b);
    let result_vals = extract_vals(&ctx, &result);

    // Reference schoolbook multiply mod (X^256 + 1)
    let mut expected = [0u32; 256];
    for i in 0..256 {
        for j in 0..256 {
            let idx = i + j;
            let val = (a_vals[i] as u64 * b_vals[j] as u64) % Q as u64;
            if idx < 256 {
                expected[idx] = (expected[idx] + val as u32) % Q;
            } else {
                expected[idx - 256] = (expected[idx - 256] + Q - val as u32) % Q;
            }
        }
    }

    for i in 0..N {
        assert_eq!(
            result_vals[i] % Q,
            expected[i],
            "poly_mul mismatch at {i}: got {}, expected {}",
            result_vals[i] % Q,
            expected[i]
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_inner_product_ntt() {
    let mut ctx = Context::<QM31>::default();

    // k=2 vector inner product
    let a0_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 3 + 1) % Q);
    let a1_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 7 + 2) % Q);
    let b0_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 11 + 3) % Q);
    let b1_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 13 + 5) % Q);

    let mut a0 = make_poly(&mut ctx, &a0_vals);
    let mut a1 = make_poly(&mut ctx, &a1_vals);
    let mut b0 = make_poly(&mut ctx, &b0_vals);
    let mut b1 = make_poly(&mut ctx, &b1_vals);

    ntt(&mut ctx, &mut a0);
    ntt(&mut ctx, &mut a1);
    ntt(&mut ctx, &mut b0);
    ntt(&mut ctx, &mut b1);

    let _result = inner_product_ntt(&mut ctx, &[a0, a1], &[b0, b1]);

    // Just verify it's a valid circuit
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}
