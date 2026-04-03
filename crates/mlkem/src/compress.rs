use circuits::context::{Context, Var};
use circuits::extract_bits::extract_bits;
use circuits::ivalue::IValue;
use circuits::ops::{eq, guess, pointwise_mul};
use circuits::simd::Simd;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::constants::Q;
use crate::zq::{ZqVar, ZqWitness};

#[cfg(test)]
#[path = "compress_test.rs"]
pub mod test;

/// Compress: round(2^d / Q * x) mod 2^d
///
/// Given x in [0, Q), computes y = round(x * 2^d / Q) mod 2^d.
/// This is: y = floor((x * 2^d + Q/2) / Q) mod 2^d.
///
/// Circuit: prover guesses y, circuit constrains:
///   x * 2^d + Q/2 = y * Q + r  where 0 <= r < Q
///   0 <= y < 2^d
pub fn compress<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    x: ZqVar,
    d: u32,
) -> Var {
    let two_d = 1u32 << d;
    let half_q = Q / 2;

    let x_int = ctx.get(x.0).raw_u32();
    let numerator = x_int as u64 * two_d as u64 + half_q as u64;
    let y_full = (numerator / Q as u64) as u32;
    let y_int = y_full % two_d;
    let t_int = y_full / two_d; // 0 or 1 (wrap-around indicator)
    let r_int = (numerator - y_full as u64 * Q as u64) as u32;

    let y_var = guess(ctx, V::from_u32(y_int));
    let r_var = guess(ctx, V::from_u32(r_int));
    let t_var = guess(ctx, V::from_u32(t_int));

    // Assert t is boolean: t * t == t
    let t_sq = pointwise_mul(ctx, t_var, t_var);
    eq(ctx, t_sq, t_var);

    // Constrain: x * 2^d + half_q = (y + t * 2^d) * Q + r
    //          = y*Q + t*2^d*Q + r
    let two_d_const = ctx.constant(QM31::from(M31::from(two_d)));
    let half_q_const = ctx.constant(QM31::from(M31::from(half_q)));
    let q_const = ctx.constant(QM31::from(M31::from(Q)));
    let two_d_q_const = ctx.constant(QM31::from(M31::from(two_d * Q)));

    let x_scaled = pointwise_mul(ctx, x.0, two_d_const);
    let lhs = circuits::eval!(ctx, (x_scaled) + (half_q_const));

    let y_q = pointwise_mul(ctx, y_var, q_const);
    let t_wrap = pointwise_mul(ctx, t_var, two_d_q_const);
    let yt = circuits::eval!(ctx, (y_q) + (t_wrap));
    let rhs = circuits::eval!(ctx, (yt) + (r_var));
    eq(ctx, lhs, rhs);

    // Range check: y < 2^d
    let y_simd = Simd::from_packed(vec![y_var], 1);
    let _y_bits = extract_bits(ctx, &y_simd, d);

    // Range check: r < Q (12 bits: 2^12 = 4096 > 3329)
    let r_simd = Simd::from_packed(vec![r_var], 1);
    let _r_bits = extract_bits(ctx, &r_simd, 12);

    y_var
}

/// Compress 4 values simultaneously, batching range checks via Simd(4).
pub fn compress_batch4<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    xs: [ZqVar; 4],
    d: u32,
) -> [Var; 4] {
    use circuits::wrappers::M31Wrapper;

    let two_d = 1u32 << d;
    let half_q = Q / 2;
    let two_d_const = ctx.constant(QM31::from(M31::from(two_d)));
    let half_q_const = ctx.constant(QM31::from(M31::from(half_q)));
    let q_const = ctx.constant(QM31::from(M31::from(Q)));
    let two_d_q_const = ctx.constant(QM31::from(M31::from(two_d * Q)));

    let mut y_vars = [ctx.zero(); 4];
    let mut r_vars = [ctx.zero(); 4];
    let mut t_vars = [ctx.zero(); 4];

    for idx in 0..4 {
        let x_int = ctx.get(xs[idx].0).raw_u32();
        let numerator = x_int as u64 * two_d as u64 + half_q as u64;
        let y_full = (numerator / Q as u64) as u32;
        let y_int = y_full % two_d;
        let t_int = y_full / two_d;
        let r_int = (numerator - y_full as u64 * Q as u64) as u32;

        y_vars[idx] = guess(ctx, V::from_u32(y_int));
        r_vars[idx] = guess(ctx, V::from_u32(r_int));
        t_vars[idx] = guess(ctx, V::from_u32(t_int));

        // Boolean check on t
        let t_sq = pointwise_mul(ctx, t_vars[idx], t_vars[idx]);
        eq(ctx, t_sq, t_vars[idx]);

        // Constraint: x * 2^d + Q/2 = y*Q + t*2^d*Q + r
        let x_scaled = pointwise_mul(ctx, xs[idx].0, two_d_const);
        let lhs = circuits::eval!(ctx, (x_scaled) + (half_q_const));
        let y_q = pointwise_mul(ctx, y_vars[idx], q_const);
        let t_wrap = pointwise_mul(ctx, t_vars[idx], two_d_q_const);
        let yt = circuits::eval!(ctx, (y_q) + (t_wrap));
        let rhs = circuits::eval!(ctx, (yt) + (r_vars[idx]));
        eq(ctx, lhs, rhs);
    }

    // Batch range check y < 2^d
    let y_wrapped: Vec<M31Wrapper<Var>> =
        y_vars.iter().map(|v| M31Wrapper::new_unsafe(*v)).collect();
    let y_simd = Simd::pack(ctx, &y_wrapped);
    let _y_bits = extract_bits(ctx, &y_simd, d);

    // Batch range check r < Q (12 bits)
    let r_wrapped: Vec<M31Wrapper<Var>> =
        r_vars.iter().map(|v| M31Wrapper::new_unsafe(*v)).collect();
    let r_simd = Simd::pack(ctx, &r_wrapped);
    let _r_bits = extract_bits(ctx, &r_simd, 12);

    y_vars
}

/// Decompress: round(Q / 2^d * y)
///
/// Given y in [0, 2^d), computes x = round(y * Q / 2^d).
/// This is: x = floor((y * Q + 2^(d-1)) / 2^d).
///
/// Circuit: prover guesses x and r, constrains:
///   y * Q + 2^(d-1) = x * 2^d + r  where 0 <= r < 2^d
///   0 <= x < Q
pub fn decompress<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    y: Var,
    d: u32,
) -> ZqVar {
    let two_d = 1u32 << d;
    let half_two_d = 1u32 << (d - 1);

    let y_int = ctx.get(y).raw_u32();
    let numerator = y_int as u64 * Q as u64 + half_two_d as u64;
    let x_int = (numerator / two_d as u64) as u32;
    let r_int = (numerator % two_d as u64) as u32;

    let x_var = guess(ctx, V::from_u32(x_int));
    let r_var = guess(ctx, V::from_u32(r_int));

    // Constrain: y * Q + 2^(d-1) = x * 2^d + r
    let q_const = ctx.constant(QM31::from(M31::from(Q)));
    let half_const = ctx.constant(QM31::from(M31::from(half_two_d)));
    let two_d_const = ctx.constant(QM31::from(M31::from(two_d)));

    let y_q = pointwise_mul(ctx, y, q_const);
    let lhs = circuits::eval!(ctx, (y_q) + (half_const));
    let x_two_d = pointwise_mul(ctx, x_var, two_d_const);
    let rhs = circuits::eval!(ctx, (x_two_d) + (r_var));
    eq(ctx, lhs, rhs);

    // Range check: x < Q (12 bits)
    let x_simd = Simd::from_packed(vec![x_var], 1);
    let _x_bits = extract_bits(ctx, &x_simd, 12);

    // Range check: r < 2^d
    let r_simd = Simd::from_packed(vec![r_var], 1);
    let _r_bits = extract_bits(ctx, &r_simd, d);

    ZqVar(x_var)
}
