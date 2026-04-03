use circuits::context::{Context, Var};
use circuits::extract_bits::extract_bits;
use circuits::ivalue::IValue;
use circuits::simd::Simd;

use crate::constants::N;
use crate::ntt::RqPoly;
use crate::zq::{ZqVar, ZqWitness, zq_sub};

#[cfg(test)]
#[path = "sampling_test.rs"]
pub mod test;

/// CBD_η: Centered Binomial Distribution sampling.
///
/// Given a byte stream (as Vars holding M31 values in [0, 256)),
/// produces a polynomial with coefficients in [-η, η] (represented mod Q).
///
/// For ML-KEM-768: η₁ = η₂ = 2, needing 64*eta bytes per polynomial.
/// Each coefficient is computed as:
///   x = (sum of η bits from stream a) - (sum of η bits from stream b)
pub fn cbd<V: IValue + ZqWitness>(
    ctx: &mut Context<V>,
    bytes: &[Var],
    eta: u32,
) -> RqPoly {
    assert_eq!(bytes.len(), (N as u32 * eta / 4) as usize);

    // Extract all bits from the byte stream
    let mut all_bits = Vec::new();
    for byte_var in bytes {
        let simd = Simd::from_packed(vec![*byte_var], 1);
        let bits = extract_bits(ctx, &simd, 8);
        all_bits.extend(bits);
    }

    // Each coefficient uses 2*eta bits
    let mut coeffs = [ZqVar(ctx.zero()); N];
    for i in 0..N {
        let bit_offset = i * 2 * eta as usize;

        // Sum first η bits (the "a" part)
        let mut sum_a = ctx.zero();
        for j in 0..eta as usize {
            let bit_var = Simd::unpack(ctx, &all_bits[bit_offset + j])[0];
            sum_a = circuits::eval!(ctx, (sum_a) + (bit_var));
        }

        // Sum next η bits (the "b" part)
        let mut sum_b = ctx.zero();
        for j in 0..eta as usize {
            let bit_var = Simd::unpack(ctx, &all_bits[bit_offset + eta as usize + j])[0];
            sum_b = circuits::eval!(ctx, (sum_b) + (bit_var));
        }

        // coefficient = sum_a - sum_b mod Q
        coeffs[i] = zq_sub(ctx, ZqVar(sum_a), ZqVar(sum_b));
    }

    RqPoly { coeffs }
}
