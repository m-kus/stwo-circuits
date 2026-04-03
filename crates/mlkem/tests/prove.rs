use std::time::Instant;

use circuit_air::statement::{INTERACTION_POW_BITS, all_circuit_components};
use circuit_common::finalize::finalize_context;
use circuit_common::preprocessed::PreprocessedCircuit;
use circuit_prover::prover::{
    BaseColumnPool, SimdBackend, prepare_circuit_proof_for_circuit_verifier,
    prove_circuit_assignment,
};
use circuit_serialize::serialize::CircuitSerialize;
use circuits::context::Context;
use circuits::ops::guess;
use circuits_stark_verifier::proof::ProofConfig;
use stwo::core::fields::m31::M31;
use stwo::core::fields::qm31::QM31;
use stwo::core::pcs::PcsConfig;

use mlkem::constants::N;
use mlkem::kem::{MLKEM_512, decaps, decrypt, encaps, encrypt};
use mlkem::ntt::{RqPoly, ntt};
use mlkem::zq::{zq_witness, ZqVar};

fn make_zero_poly(ctx: &mut Context<QM31>) -> RqPoly {
    let coeffs: [ZqVar; 256] = std::array::from_fn(|_| zq_witness(ctx, 0));
    RqPoly { coeffs }
}

fn make_zero_bytes(ctx: &mut Context<QM31>, n: usize) -> Vec<circuits::context::Var> {
    (0..n)
        .map(|_| guess(ctx, QM31::from(M31::from(0u32))))
        .collect()
}

/// Shared prove-and-measure logic.
fn prove_and_measure(ctx: &mut Context<QM31>, label: &str) {
    let t0 = Instant::now();

    eprintln!("\n=== {label} ===");
    eprintln!("Gate stats: {:?}", ctx.stats);

    let t_fin = Instant::now();
    ctx.finalize_guessed_vars();
    finalize_context(ctx);
    ctx.validate_circuit();
    let finalize_time = t_fin.elapsed();
    eprintln!("Finalize + validate: {finalize_time:.2?}");

    let t_pre = Instant::now();
    let preprocessed = PreprocessedCircuit::preprocess_circuit(ctx);
    let preprocess_time = t_pre.elapsed();
    eprintln!("Preprocess: {preprocess_time:.2?}");
    eprintln!("Trace log size: {}", preprocessed.params.trace_log_size);

    let pcs_config = PcsConfig::default();
    let t_prove = Instant::now();
    let circuit_proof = prove_circuit_assignment(
        ctx.values(),
        &preprocessed,
        &BaseColumnPool::<SimdBackend>::new(),
        pcs_config,
    );
    let prove_time = t_prove.elapsed();
    eprintln!("Prove: {prove_time:.2?}");

    assert!(
        circuit_proof.stark_proof.is_ok(),
        "Proof failed: {}",
        circuit_proof.stark_proof.err().unwrap()
    );

    // Proof size
    let ids = preprocessed.preprocessed_trace.ids();
    let proof_config = ProofConfig::from_components(
        &all_circuit_components::<QM31>(),
        ids.len(),
        &circuit_proof.pcs_config,
        INTERACTION_POW_BITS,
    );
    let (proof, _) = prepare_circuit_proof_for_circuit_verifier(circuit_proof, &proof_config);
    let mut serialized = vec![];
    proof.serialize(&mut serialized);

    let total = t0.elapsed();
    eprintln!("Proof size: {} bytes ({:.1} KB)", serialized.len(), serialized.len() as f64 / 1024.0);
    eprintln!("Total: {total:.2?} (finalize {finalize_time:.2?} + preprocess {preprocess_time:.2?} + prove {prove_time:.2?})");
}

/// Benchmark: encrypt + decrypt (no hashing).
#[test]
fn bench_encrypt_decrypt() {
    let t_start = Instant::now();
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    let a_hat: Vec<Vec<RqPoly>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let mut p = if i == j {
                        let coeffs: [ZqVar; 256] = std::array::from_fn(|idx| {
                            zq_witness(&mut ctx, if idx == 0 { 1 } else { 0 })
                        });
                        RqPoly { coeffs }
                    } else {
                        make_zero_poly(&mut ctx)
                    };
                    ntt(&mut ctx, &mut p);
                    p
                })
                .collect()
        })
        .collect();

    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();

    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    let cbd_eta1 = (N as u32 * params.eta1 / 4) as usize;
    let cbd_eta2 = (N as u32 * params.eta2 / 4) as usize;
    let r_bytes: Vec<Vec<_>> = (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_eta1)).collect();
    let e1_bytes: Vec<Vec<_>> = (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_eta2)).collect();
    let e2_bytes = make_zero_bytes(&mut ctx, cbd_eta2);

    let (u_vec, v) = encrypt(&mut ctx, params, &a_hat, &t_hat, &message_bits, &r_bytes, &e1_bytes, &e2_bytes);
    let _recovered = decrypt(&mut ctx, &s_hat, &u_vec, &v);

    eprintln!("Circuit build: {:.2?}", t_start.elapsed());
    prove_and_measure(&mut ctx, "ML-KEM-512 Encrypt+Decrypt (no hashing)");
}

/// Benchmark: full encaps + decaps with Blake2s hashing.
#[test]
fn bench_encaps_decaps() {
    let t_start = Instant::now();
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(42u32 + i))))
        .collect();

    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();

    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    let (u_vec, v, _ss_enc) = encaps(&mut ctx, params, &rho, &t_hat, &message_bits);
    let (_msg, _ss_dec) = decaps(&mut ctx, &s_hat, &u_vec, &v, &rho);

    eprintln!("Circuit build: {:.2?}", t_start.elapsed());
    prove_and_measure(&mut ctx, "ML-KEM-512 Encaps+Decaps (with Blake2s)");
}
