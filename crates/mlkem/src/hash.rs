use circuits::blake::{HashValue, blake};
use circuits::context::{Context, Var};
use circuits::ivalue::IValue;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;

#[cfg(test)]
#[path = "hash_test.rs"]
pub mod test;

/// Hash function H: maps arbitrary-length input to a 256-bit digest.
/// Uses Blake2s over M31 words (non-standard, adapted for M31 circuit).
/// Returns (hash_lo, hash_hi) as two QM31 variables = 8 M31 words = 256 bits.
pub fn hash_h<V: IValue>(
    ctx: &mut Context<V>,
    input: &[Var],
    n_bytes: usize,
) -> HashValue<Var> {
    blake(ctx, input, n_bytes)
}

/// Hash function G: maps input to a 512-bit digest.
/// Implemented as two Blake2s calls with domain separation.
/// Returns (hash_0, hash_1, hash_2, hash_3) — four QM31 variables = 16 M31 words.
pub fn hash_g<V: IValue>(
    ctx: &mut Context<V>,
    input: &[Var],
    n_bytes: usize,
) -> (HashValue<Var>, HashValue<Var>) {
    // Domain-separate by prepending a tag byte.
    // Tag 0 for first half, tag 1 for second half.
    let tag0 = ctx.constant(QM31::from(M31::from(0u32)));
    let tag1 = ctx.constant(QM31::from(M31::from(1u32)));

    let mut input0 = vec![tag0];
    input0.extend_from_slice(input);
    let mut input1 = vec![tag1];
    input1.extend_from_slice(input);

    let h0 = blake(ctx, &input0, n_bytes + 16);
    let h1 = blake(ctx, &input1, n_bytes + 16);
    (h0, h1)
}

/// PRF: keyed pseudorandom function.
/// PRF(key, nonce) = Blake2s(key || nonce).
/// Key is given as Vars, nonce as a u8 constant.
pub fn prf<V: IValue>(
    ctx: &mut Context<V>,
    key: &[Var],
    key_bytes: usize,
    nonce: u8,
) -> HashValue<Var> {
    let nonce_var = ctx.constant(QM31::from(M31::from(nonce as u32)));
    let mut input = key.to_vec();
    input.push(nonce_var);
    // key_bytes + 1 byte for nonce (padded to 4 bytes in the M31 word)
    blake(ctx, &input, key_bytes + 4)
}

/// XOF (Extendable Output Function): generates a stream of output.
/// Implemented by calling Blake2s repeatedly with an incrementing counter.
/// Each call produces 32 bytes (8 M31 words packed as 2 QM31 vars).
///
/// Returns `n_blocks` hash outputs.
pub fn xof<V: IValue>(
    ctx: &mut Context<V>,
    seed: &[Var],
    seed_bytes: usize,
    n_blocks: usize,
) -> Vec<HashValue<Var>> {
    (0..n_blocks)
        .map(|i| {
            let counter = ctx.constant(QM31::from(M31::from(i as u32)));
            let mut input = seed.to_vec();
            input.push(counter);
            blake(ctx, &input, seed_bytes + 4)
        })
        .collect()
}
