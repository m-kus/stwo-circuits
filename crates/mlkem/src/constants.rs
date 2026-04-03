/// ML-KEM field modulus.
pub const Q: u32 = 3329;

/// Polynomial degree.
pub const N: usize = 256;

/// Lazy reduction bound (2 * Q).
pub const TWO_Q: u32 = 2 * Q;

/// 128^{-1} mod Q, used for inverse NTT scaling.
/// The ML-KEM NTT maps 256 coefficients to 128 degree-1 polynomial pairs,
/// so the effective transform size is 128.
pub const NTT_SCALE_INV: u32 = pow_mod(128, Q - 2, Q);

/// Primitive 256th root of unity mod Q. Verified: 17^256 ≡ 1 (mod 3329).
const ZETA: u32 = 17;

/// NTT twiddle factors: ZETAS\[k\] = ζ^{BitRev7(k)} mod Q for k = 0..128.
/// Index 0 is unused by the NTT loop itself but used by basemul.
pub const ZETAS: [u32; 128] = compute_zetas();

/// Inverse of each zeta: ZETAS_INV\[k\] = ζ^{-(BitRev7(k))} mod Q for k = 0..128.
/// Note: The InvNTT uses ZETAS (same as forward), not ZETAS_INV.
/// This array is kept for basemul and other operations that need modular inverses.
pub const ZETAS_INV: [u32; 128] = compute_zetas_inv();

const fn pow_mod(mut base: u32, mut exp: u32, modulus: u32) -> u32 {
    let mut result: u64 = 1;
    let m = modulus as u64;
    base = base % modulus;
    let mut b = base as u64;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * b) % m;
        }
        exp /= 2;
        b = (b * b) % m;
    }
    result as u32
}

const fn bit_rev7(x: u32) -> u32 {
    let mut result = 0u32;
    let mut v = x;
    let mut i = 0;
    while i < 7 {
        result = (result << 1) | (v & 1);
        v >>= 1;
        i += 1;
    }
    result
}

const fn compute_zetas() -> [u32; 128] {
    let mut zetas = [0u32; 128];
    let mut k = 0;
    while k < 128 {
        zetas[k] = pow_mod(ZETA, bit_rev7(k as u32), Q);
        k += 1;
    }
    zetas
}

const fn compute_zetas_inv() -> [u32; 128] {
    let mut zetas_inv = [0u32; 128];
    let mut k = 0;
    while k < 128 {
        // ζ^{-BitRev7(k)} = ζ^{256 - BitRev7(k)} since ζ^256 = 1
        let exp = 256 - bit_rev7(k as u32);
        zetas_inv[k] = pow_mod(ZETA, exp, Q);
        k += 1;
    }
    zetas_inv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeta_is_primitive_root() {
        assert_eq!(pow_mod(ZETA, 256, Q), 1);
        assert_ne!(pow_mod(ZETA, 128, Q), 1);
    }

    #[test]
    fn test_ntt_scale_inv() {
        assert_eq!(((NTT_SCALE_INV as u64 * 128) % Q as u64) as u32, 1);
    }

    #[test]
    fn test_zetas_first_values() {
        // ZETAS[0] = ζ^{BitRev7(0)} = ζ^0 = 1
        assert_eq!(ZETAS[0], 1);
        // ZETAS[1] = ζ^{BitRev7(1)} = ζ^64
        assert_eq!(ZETAS[1], pow_mod(ZETA, 64, Q));
    }

    #[test]
    fn test_zetas_inv_are_inverses() {
        for k in 0..128 {
            let prod = (ZETAS[k] as u64 * ZETAS_INV[k] as u64) % Q as u64;
            assert_eq!(prod, 1, "ZETAS[{k}] * ZETAS_INV[{k}] != 1 mod Q");
        }
    }
}
