use circuits::context::Context;
use stwo::core::fields::qm31::QM31;

use crate::constants::{N, Q, ZETAS};
use crate::ntt::*;
use crate::zq::{zq_witness, ZqVar};

/// Reference (native) NTT for test vectors — operates on plain u32 arrays.
fn ref_ntt(f: &mut [u32; 256]) {
    let mut k = 1usize;
    let mut len = 128;
    while len >= 2 {
        let mut start = 0;
        while start < 256 {
            let zeta = ZETAS[k] as u64;
            k += 1;
            for j in start..(start + len) {
                let t = ((zeta * f[j + len] as u64) % Q as u64) as u32;
                f[j + len] = (f[j] + Q - t) % Q;
                f[j] = (f[j] + t) % Q;
            }
            start += 2 * len;
        }
        len >>= 1;
    }
}

/// Reference (native) inverse NTT — uses SAME zetas as forward, reversed k.
fn ref_inv_ntt(f: &mut [u32; 256]) {
    let mut k = 127usize;
    let mut len = 2;
    while len <= 128 {
        let mut start = 0;
        while start < 256 {
            let zeta = ZETAS[k] as u64;
            k = k.wrapping_sub(1);
            for j in start..(start + len) {
                let t = f[j];
                f[j] = (t + f[j + len]) % Q;
                f[j + len] = ((zeta * ((f[j + len] + Q - t) as u64)) % Q as u64) as u32;
            }
            start += 2 * len;
        }
        len <<= 1;
    }
    let n_inv = 3303u64; // 128^{-1} mod Q
    for coeff in f.iter_mut() {
        *coeff = ((n_inv * *coeff as u64) % Q as u64) as u32;
    }
}

#[test]
fn test_ref_roundtrip() {
    let original: [u32; 256] = std::array::from_fn(|i| ((i * 137 + 42) % Q as usize) as u32);
    let mut f = original;
    ref_ntt(&mut f);
    ref_inv_ntt(&mut f);
    for i in 0..256 {
        assert_eq!(f[i], original[i], "ref roundtrip failed at index {i}: got {}, expected {}", f[i], original[i]);
    }
}

fn make_poly(ctx: &mut Context<QM31>, vals: &[u32; 256]) -> RqPoly {
    let coeffs: [ZqVar; 256] = std::array::from_fn(|i| zq_witness(ctx, vals[i]));
    RqPoly { coeffs }
}

fn extract_vals(ctx: &Context<QM31>, poly: &RqPoly) -> [u32; 256] {
    std::array::from_fn(|i| ctx.get(poly.coeffs[i].0).0 .0 .0)
}

#[test]
fn test_ntt_known_vector() {
    let mut ctx = Context::<QM31>::default();

    // Simple input: x = 1, all other coefficients 0
    let mut input = [0u32; 256];
    input[1] = 1;
    let mut poly = make_poly(&mut ctx, &input);

    ntt(&mut ctx, &mut poly);
    let result = extract_vals(&ctx, &poly);

    // Compare with reference
    let mut expected = input;
    ref_ntt(&mut expected);

    // The NTT result values may be in [0, 2Q) due to lazy reduction.
    // Compare mod Q.
    for i in 0..N {
        assert_eq!(
            result[i] % Q,
            expected[i] % Q,
            "NTT mismatch at index {i}: got {}, expected {}",
            result[i],
            expected[i]
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_inv_ntt_known_vector() {
    let mut ctx = Context::<QM31>::default();

    // Input in NTT domain: all 1s
    let input = [1u32; 256];
    let mut poly = make_poly(&mut ctx, &input);

    inv_ntt(&mut ctx, &mut poly);
    let result = extract_vals(&ctx, &poly);

    let mut expected = input;
    ref_inv_ntt(&mut expected);

    for i in 0..N {
        assert_eq!(
            result[i] % Q,
            expected[i] % Q,
            "InvNTT mismatch at index {i}: got {}, expected {}",
            result[i],
            expected[i]
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_ntt_inverse_roundtrip() {
    let mut ctx = Context::<QM31>::default();

    // Random-ish input
    let input: [u32; 256] = std::array::from_fn(|i| ((i * 137 + 42) % Q as usize) as u32);
    let mut poly = make_poly(&mut ctx, &input);

    ntt(&mut ctx, &mut poly);
    inv_ntt(&mut ctx, &mut poly);
    let result = extract_vals(&ctx, &poly);

    for i in 0..N {
        assert_eq!(
            result[i] % Q,
            input[i],
            "Roundtrip mismatch at index {i}: got {}, expected {}",
            result[i] % Q,
            input[i]
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_ntt_linearity() {
    let mut ctx = Context::<QM31>::default();

    let a_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 7 + 3) % Q);
    let b_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 13 + 11) % Q);
    let sum_vals: [u32; 256] = std::array::from_fn(|i| (a_vals[i] + b_vals[i]) % Q);

    // NTT(a + b)
    let mut sum_ref = sum_vals;
    ref_ntt(&mut sum_ref);

    // NTT(a) + NTT(b)
    let mut a_ref = a_vals;
    let mut b_ref = b_vals;
    ref_ntt(&mut a_ref);
    ref_ntt(&mut b_ref);
    let sum_parts: [u32; 256] = std::array::from_fn(|i| (a_ref[i] + b_ref[i]) % Q);

    // Both should match
    for i in 0..N {
        assert_eq!(sum_ref[i], sum_parts[i], "Linearity failed at index {i}");
    }

    // Now test the circuit NTT on the sum
    let mut poly = make_poly(&mut ctx, &sum_vals);
    ntt(&mut ctx, &mut poly);
    let circuit_result = extract_vals(&ctx, &poly);

    for i in 0..N {
        assert_eq!(
            circuit_result[i] % Q,
            sum_ref[i],
            "Circuit NTT linearity mismatch at index {i}"
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_pointwise_mul() {
    let mut ctx = Context::<QM31>::default();

    let a_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 3 + 1) % Q);
    let b_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 5 + 2) % Q);

    // Convert to NTT domain
    let mut a_ntt = a_vals;
    let mut b_ntt = b_vals;
    ref_ntt(&mut a_ntt);
    ref_ntt(&mut b_ntt);

    let a_poly = make_poly(&mut ctx, &a_ntt);
    let b_poly = make_poly(&mut ctx, &b_ntt);

    let result = ntt_pointwise_mul(&mut ctx, &a_poly, &b_poly);
    let result_vals = extract_vals(&ctx, &result);

    // Compute reference: multiply in coefficient domain, then NTT
    let mut prod_ref = [0u32; 256];
    // Schoolbook multiply mod (X^256 + 1)
    for i in 0..256 {
        for j in 0..256 {
            let idx = i + j;
            let val = (a_vals[i] as u64 * b_vals[j] as u64) % Q as u64;
            if idx < 256 {
                prod_ref[idx] = (prod_ref[idx] + val as u32) % Q;
            } else {
                // X^256 = -1 mod (X^256 + 1)
                prod_ref[idx - 256] = (prod_ref[idx - 256] + Q - val as u32) % Q;
            }
        }
    }
    ref_ntt(&mut prod_ref);

    // Compare NTT(a*b) with pointwise(NTT(a), NTT(b))
    for i in 0..N {
        assert_eq!(
            result_vals[i] % Q,
            prod_ref[i],
            "Pointwise mul mismatch at index {i}: got {}, expected {}",
            result_vals[i] % Q,
            prod_ref[i]
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_poly_mul_roundtrip() {
    let mut ctx = Context::<QM31>::default();

    let a_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 3 + 1) % Q);
    let b_vals: [u32; 256] = std::array::from_fn(|i| (i as u32 * 5 + 2) % Q);

    let mut a_poly = make_poly(&mut ctx, &a_vals);
    let mut b_poly = make_poly(&mut ctx, &b_vals);

    let result = poly_mul(&mut ctx, &mut a_poly, &mut b_poly);
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
            "poly_mul mismatch at index {i}: got {}, expected {}",
            result_vals[i] % Q,
            expected[i]
        );
    }

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}

#[test]
fn test_ntt_gate_counts() {
    let mut ctx = Context::<QM31>::default();
    let input = [0u32; 256];
    let mut poly = make_poly(&mut ctx, &input);

    ntt(&mut ctx, &mut poly);
    let stats = &ctx.stats;
    eprintln!("Forward NTT gate stats: {stats:?}");

    ctx.finalize_guessed_vars();
    ctx.validate_circuit();
}
