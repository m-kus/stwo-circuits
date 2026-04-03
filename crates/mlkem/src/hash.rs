use circuits::blake::{HashValue, blake};
use circuits::context::{Context, Var};
use circuits::ivalue::IValue;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

#[cfg(test)]
#[path = "hash_test.rs"]
pub mod test;

/// Bytes per QM31 variable (4 M31 coordinates × 4 bytes each).
const BYTES_PER_QM31: usize = 16;

/// Hash function H: Blake2s hash of QM31 vars.
/// Returns 256-bit digest as (lo, hi) QM31 pair.
pub fn hash_h<V: IValue>(ctx: &mut Context<V>, input: &[Var]) -> HashValue<Var> {
    blake(ctx, input, input.len() * BYTES_PER_QM31)
}

/// Hash function G: produces 512-bit output via two domain-separated Blake2s calls.
/// Returns two HashValues (4 QM31 = 512 bits total).
pub fn hash_g<V: IValue>(
    ctx: &mut Context<V>,
    input: &[Var],
) -> (HashValue<Var>, HashValue<Var>) {
    let tag0 = ctx.constant(QM31::from(M31::from(0u32)));
    let tag1 = ctx.constant(QM31::from(M31::from(1u32)));

    let mut input0 = vec![tag0];
    input0.extend_from_slice(input);
    let mut input1 = vec![tag1];
    input1.extend_from_slice(input);

    let h0 = blake(ctx, &input0, input0.len() * BYTES_PER_QM31);
    let h1 = blake(ctx, &input1, input1.len() * BYTES_PER_QM31);
    (h0, h1)
}

/// PRF: keyed pseudorandom function.
/// PRF(key, nonce) = Blake2s(key || nonce_var).
pub fn prf<V: IValue>(ctx: &mut Context<V>, key: &[Var], nonce: u8) -> HashValue<Var> {
    let nonce_var = ctx.constant(QM31::from(M31::from(nonce as u32)));
    let mut input = key.to_vec();
    input.push(nonce_var);
    blake(ctx, &input, input.len() * BYTES_PER_QM31)
}

/// XOF: extendable output via repeated Blake2s with incrementing counter.
/// Each call produces 32 bytes (2 QM31).
pub fn xof<V: IValue>(
    ctx: &mut Context<V>,
    seed: &[Var],
    n_blocks: usize,
) -> Vec<HashValue<Var>> {
    (0..n_blocks)
        .map(|i| {
            let counter = ctx.constant(QM31::from(M31::from(i as u32)));
            let mut input = seed.to_vec();
            input.push(counter);
            blake(ctx, &input, input.len() * BYTES_PER_QM31)
        })
        .collect()
}
