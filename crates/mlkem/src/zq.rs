use circuits::context::{Context, Var};
use circuits::extract_bits::extract_bits;
use circuits::ivalue::{IValue, NoValue};
use circuits::ops::{eq, guess, pointwise_mul};
use circuits::simd::Simd;
use circuits::wrappers::M31Wrapper;
use circuits::eval;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

use crate::constants::Q;

#[cfg(test)]
#[path = "zq_test.rs"]
pub mod test;

/// Extension trait for computing integer division/modulo by Q on circuit witness values.
///
/// For `QM31`: extracts the first M31 coordinate and computes integer div/mod.
/// For `NoValue`: returns dummy values (circuit topology is the same either way).
pub trait ZqWitness: IValue {
    fn int_divmod_q(self) -> (Self, Self);
    fn int_ge_q(self) -> Self;
    /// Extract the first M31 coordinate as a u32 (for witness computation).
    /// Returns 0 for NoValue.
    fn raw_u32(self) -> u32;
    /// Create a value from a u32 (wraps into M31 → QM31).
    fn from_u32(v: u32) -> Self;
}

impl ZqWitness for QM31 {
    fn int_divmod_q(self) -> (Self, Self) {
        let x = self.0 .0 .0;
        let q = x / Q;
        let r = x % Q;
        (QM31::from(M31::from(q)), QM31::from(M31::from(r)))
    }

    fn int_ge_q(self) -> Self {
        let x = self.0 .0 .0;
        QM31::from(M31::from(if x >= Q { 1u32 } else { 0 }))
    }

    fn raw_u32(self) -> u32 {
        self.0 .0 .0
    }

    fn from_u32(v: u32) -> Self {
        QM31::from(M31::from(v))
    }
}

impl ZqWitness for NoValue {
    fn int_divmod_q(self) -> (Self, Self) {
        (NoValue, NoValue)
    }

    fn int_ge_q(self) -> Self {
        NoValue
    }

    fn raw_u32(self) -> u32 {
        0
    }

    fn from_u32(_v: u32) -> Self {
        NoValue
    }
}

/// A circuit variable known to hold a value in [0, 2Q) where Q = 3329.
/// The lazy reduction bound 2Q = 6658 < 8192 = 2^13, so 13-bit range checks suffice.
#[derive(Clone, Copy, Debug)]
pub struct ZqVar(pub Var);

/// Reduce a circuit variable with integer value < 2^`input_bits` to [0, Q).
///
/// The prover guesses quotient q and remainder r such that x = q*Q + r as integers.
/// The circuit constrains this equation (valid in M31 since all values < P) and
/// range-checks both q and r via bit decomposition.
pub fn mod_reduce_lazy<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    x: Var,
    input_bits: u32,
) -> ZqVar {
    let (q_val, r_val) = ctx.get(x).int_divmod_q();

    let q_var = guess(ctx, q_val);
    let r_var = guess(ctx, r_val);

    // Constrain: q * Q + r = x
    let q_const = ctx.constant(QM31::from(M31::from(Q)));
    let q_times_q = pointwise_mul(ctx, q_var, q_const);
    let reconstructed = eval!(ctx, (q_times_q) + (r_var));
    eq(ctx, reconstructed, x);

    // Range check r < 2^13 (8192 > 2Q = 6658, so this implies r < 2Q)
    let r_simd = Simd::from_packed(vec![r_var], 1);
    let _r_bits = extract_bits(ctx, &r_simd, 13);

    // Range check q: q < x / Q < 2^input_bits / Q < 2^(input_bits - 11)
    let q_bits = input_bits.saturating_sub(11).max(1);
    let q_simd = Simd::from_packed(vec![q_var], 1);
    let _q_bits = extract_bits(ctx, &q_simd, q_bits);

    ZqVar(r_var)
}

/// Create a ZqVar from a constant value in [0, Q). No range check needed.
pub fn zq_constant<V: IValue>(ctx: &mut Context<V>, value: u32) -> ZqVar {
    assert!(value < Q, "zq_constant: value {value} >= Q");
    ZqVar(ctx.constant(QM31::from(M31::from(value))))
}

/// Create a ZqVar from a witness value in [0, Q), with a 13-bit range check.
pub fn zq_witness<V: IValue + ZqWitness>(ctx: &mut Context<V>, value: u32) -> ZqVar {
    assert!(value < Q, "zq_witness: value {value} >= Q");
    let var = guess(ctx, V::from_qm31(QM31::from(M31::from(value))));
    let simd = Simd::from_packed(vec![var], 1);
    let _bits = extract_bits(ctx, &simd, 13);
    ZqVar(var)
}

/// Multiply two ZqVars: a, b in [0, 2Q), product in [0, 4Q^2) < 2^26.
/// Returns a reduced value in [0, Q).
pub fn zq_mul<V: IValue + ZqWitness>(ctx: &mut Context<V>, a: ZqVar, b: ZqVar) -> ZqVar {
    let prod = pointwise_mul(ctx, a.0, b.0);
    mod_reduce_lazy(ctx, prod, 26)
}

/// Add two ZqVars: a, b in [0, 2Q), sum in [0, 4Q) < 2^14.
///
/// Returns unreduced. The result is valid for one subsequent multiplication
/// (4Q * 2Q = 8Q^2 < 2^27 < P).
pub fn zq_add<V: IValue>(ctx: &mut Context<V>, a: ZqVar, b: ZqVar) -> ZqVar {
    ZqVar(eval!(ctx, (a.0) + (b.0)))
}

/// Subtract two ZqVars: a, b in [0, 2Q).
/// Adds 2Q offset to ensure non-negative, then reduces.
/// Result in [0, Q).
pub fn zq_sub<V: IValue + ZqWitness>(ctx: &mut Context<V>, a: ZqVar, b: ZqVar) -> ZqVar {
    let offset = ctx.constant(QM31::from(M31::from(2 * Q)));
    let diff = eval!(ctx, ((a.0) + (offset)) - (b.0));
    // diff in [0, 4Q) < 2^14
    mod_reduce_lazy(ctx, diff, 14)
}

/// Multiply a ZqVar by a constant twiddle factor c in [0, Q).
/// a in [0, 2Q), c < Q, product < 2Q^2 < 2^25.
pub fn zq_mul_const<V: IValue + ZqWitness>(ctx: &mut Context<V>, a: ZqVar, c: u32) -> ZqVar {
    assert!(c < Q, "zq_mul_const: twiddle {c} >= Q");
    let c_var = ctx.constant(QM31::from(M31::from(c)));
    let prod = pointwise_mul(ctx, a.0, c_var);
    mod_reduce_lazy(ctx, prod, 25)
}

/// Strict reduction: bring a ZqVar from [0, 2Q) to [0, Q).
///
/// Guesses a boolean for whether x >= Q, subtracts Q conditionally, then
/// range-checks the result with 12 bits.
pub fn mod_reduce_strict<V: IValue + ZqWitness>(ctx: &mut Context<V>, x: ZqVar) -> ZqVar {
    let ge_q_val = ctx.get(x.0).int_ge_q();
    let ge_q_var = guess(ctx, ge_q_val);

    // Assert ge_q is boolean: ge_q^2 == ge_q
    let ge_q_sq = pointwise_mul(ctx, ge_q_var, ge_q_var);
    eq(ctx, ge_q_sq, ge_q_var);

    // r = x - ge_q * Q
    let q_const = ctx.constant(QM31::from(M31::from(Q)));
    let sub_val = pointwise_mul(ctx, ge_q_var, q_const);
    let r = eval!(ctx, (x.0) - (sub_val));

    // Range check r < 2^12 (4096 > Q-1 = 3328)
    let r_simd = Simd::from_packed(vec![r], 1);
    let _bits = extract_bits(ctx, &r_simd, 12);

    ZqVar(r)
}

/// Batch-reduce 4 values simultaneously using Simd packing.
/// Each input must have integer value < 2^`input_bits`.
/// Returns 4 ZqVars, each in [0, Q).
pub fn mod_reduce_lazy_batch4<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    vars: [Var; 4],
    input_bits: u32,
) -> [ZqVar; 4] {
    // Guess quotients and remainders
    let mut q_vars = [ctx.zero(); 4];
    let mut r_vars = [ctx.zero(); 4];
    for i in 0..4 {
        let (q_val, r_val) = ctx.get(vars[i]).int_divmod_q();
        q_vars[i] = guess(ctx, q_val);
        r_vars[i] = guess(ctx, r_val);
    }

    // Constrain: q[i] * Q + r[i] = x[i] for each
    let q_const = ctx.constant(QM31::from(M31::from(Q)));
    for i in 0..4 {
        let q_times_q = pointwise_mul(ctx, q_vars[i], q_const);
        let reconstructed = eval!(ctx, (q_times_q) + (r_vars[i]));
        eq(ctx, reconstructed, vars[i]);
    }

    // Batch range check remainders using Simd(4)
    let r_wrapped: Vec<M31Wrapper<Var>> =
        r_vars.iter().map(|v| M31Wrapper::new_unsafe(*v)).collect();
    let r_simd = Simd::pack(ctx, &r_wrapped);
    let _r_bits = extract_bits(ctx, &r_simd, 13);

    // Batch range check quotients using Simd(4)
    let q_bits = input_bits.saturating_sub(11).max(1);
    let q_wrapped: Vec<M31Wrapper<Var>> =
        q_vars.iter().map(|v| M31Wrapper::new_unsafe(*v)).collect();
    let q_simd = Simd::pack(ctx, &q_wrapped);
    let _q_bits = extract_bits(ctx, &q_simd, q_bits);

    r_vars.map(ZqVar)
}
