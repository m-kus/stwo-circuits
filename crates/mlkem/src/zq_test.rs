use circuits::context::Context;
use circuits::ops::guess;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::constants::Q;
use crate::zq::*;

fn m31_qm31(v: u32) -> QM31 {
    QM31::from(M31::from(v))
}

fn zq_val(ctx: &Context<QM31>, v: ZqVar) -> u32 {
    ctx.get(v.0).0 .0 .0
}

#[test]
fn test_mod_reduce_lazy_small() {
    let mut ctx = Context::<QM31>::default();
    // x = 100, should reduce to 100 (already < Q)
    let x = guess(&mut ctx, m31_qm31(100));
    let r = mod_reduce_lazy(&mut ctx, x, 13);

    assert_eq!(zq_val(&ctx, r), 100);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_mod_reduce_lazy_exact_q() {
    let mut ctx = Context::<QM31>::default();
    // x = Q = 3329, should reduce to 0
    let x = guess(&mut ctx, m31_qm31(Q));
    let r = mod_reduce_lazy(&mut ctx, x, 13);

    assert_eq!(zq_val(&ctx, r), 0);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_mod_reduce_lazy_large() {
    let mut ctx = Context::<QM31>::default();
    // x = 10000, should reduce to 10000 mod 3329 = 13
    let x = guess(&mut ctx, m31_qm31(10000));
    let r = mod_reduce_lazy(&mut ctx, x, 14);

    assert_eq!(zq_val(&ctx, r), 10000 % Q);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_mod_reduce_lazy_product_range() {
    let mut ctx = Context::<QM31>::default();
    // Simulate a product of two values near 2Q: (2*3329-1)^2 = 6657^2 = 44,315,649
    let x = guess(&mut ctx, m31_qm31(6657 * 6657));
    let r = mod_reduce_lazy(&mut ctx, x, 26);

    assert_eq!(zq_val(&ctx, r), (6657u32 * 6657) % Q);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_mul() {
    let mut ctx = Context::<QM31>::default();
    let a = zq_witness(&mut ctx, 1234);
    let b = zq_witness(&mut ctx, 2345);
    let c = zq_mul(&mut ctx, a, b);

    assert_eq!(zq_val(&ctx, c), (1234u32 * 2345) % Q);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_mul_const() {
    let mut ctx = Context::<QM31>::default();
    let a = zq_witness(&mut ctx, 2000);
    let c = zq_mul_const(&mut ctx, a, 3000);

    assert_eq!(zq_val(&ctx, c), (2000u32 * 3000) % Q);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_add() {
    let mut ctx = Context::<QM31>::default();
    let a = zq_witness(&mut ctx, 1000);
    let b = zq_witness(&mut ctx, 2000);
    let c = zq_add(&mut ctx, a, b);

    assert_eq!(zq_val(&ctx, c), 3000);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_sub() {
    let mut ctx = Context::<QM31>::default();
    let a = zq_witness(&mut ctx, 1000);
    let b = zq_witness(&mut ctx, 2000);
    // 1000 - 2000 mod Q = 1000 - 2000 + 3329 = 2329
    let c = zq_sub(&mut ctx, a, b);

    assert_eq!(zq_val(&ctx, c), (1000 + Q - 2000) % Q);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_sub_same() {
    let mut ctx = Context::<QM31>::default();
    let a = zq_witness(&mut ctx, 500);
    let b = zq_witness(&mut ctx, 500);
    let c = zq_sub(&mut ctx, a, b);

    assert_eq!(zq_val(&ctx, c), 0);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_mod_reduce_strict() {
    let mut ctx = Context::<QM31>::default();

    // Value already < Q: should stay the same
    let a = zq_witness(&mut ctx, 100);
    let a_strict = mod_reduce_strict(&mut ctx, a);
    assert_eq!(zq_val(&ctx, a_strict), 100);

    // Value in [Q, 2Q): should subtract Q
    // First create a value in [Q, 2Q) via addition
    let b = zq_witness(&mut ctx, 3000);
    let c = zq_witness(&mut ctx, 200);
    let sum = zq_add(&mut ctx, b, c); // 3200, < Q so still fine
    let d = zq_witness(&mut ctx, 300);
    let sum2 = zq_add(&mut ctx, sum, d); // 3500 > Q
    let strict = mod_reduce_strict(&mut ctx, sum2);
    assert_eq!(zq_val(&ctx, strict), 3500 - Q);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_mul_then_add() {
    let mut ctx = Context::<QM31>::default();

    // Multiply, then add, then multiply again — tests overflow tracking
    let a = zq_witness(&mut ctx, 1000);
    let b = zq_witness(&mut ctx, 2000);
    let c = zq_mul(&mut ctx, a, b); // (1000*2000) % Q

    let d = zq_witness(&mut ctx, 500);
    let e = zq_add(&mut ctx, c, d); // c + 500, unreduced but < 2Q + Q < 4Q

    let f = zq_witness(&mut ctx, 100);
    let g = zq_mul(&mut ctx, e, f); // should still work, product < 4Q * Q < 2^25

    let expected = ((((1000u32 * 2000) % Q) + 500) * 100) % Q;
    assert_eq!(zq_val(&ctx, g), expected);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_mod_reduce_lazy_batch4() {
    let mut ctx = Context::<QM31>::default();

    let vals = [10000u32, 20000, 5000, 3329];
    let vars = vals.map(|v| guess(&mut ctx, m31_qm31(v)));
    let results = mod_reduce_lazy_batch4(&mut ctx, vars, 15);

    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(zq_val(&ctx, results[i]), v % Q, "batch4 failed at index {i}");
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_constant() {
    let mut ctx = Context::<QM31>::default();
    let c = zq_constant(&mut ctx, 42);
    assert_eq!(zq_val(&ctx, c), 42);
    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_zq_operations_chain() {
    let mut ctx = Context::<QM31>::default();
    ctx.enable_assert_eq_on_eval();

    // Compute (a * b + c) * d mod Q
    let a = zq_witness(&mut ctx, 1111);
    let b = zq_witness(&mut ctx, 2222);
    let c = zq_witness(&mut ctx, 333);
    let d = zq_witness(&mut ctx, 444);

    let ab = zq_mul(&mut ctx, a, b);
    let ab_c = zq_add(&mut ctx, ab, c);
    let result = zq_mul(&mut ctx, ab_c, d);

    let expected = ((((1111u64 * 2222) % Q as u64) + 333) * 444 % Q as u64) as u32;
    assert_eq!(zq_val(&ctx, result), expected);

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}
