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
use mlkem::kem::{MLKEM_512, decrypt, encrypt};
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

#[test]
fn test_mlkem_prove_and_verify() {
    let t_start = Instant::now();

    // --- Build circuit ---
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    // Identity-like matrix A in NTT domain
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

    // Secret key s = 0
    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    // t_hat = 0
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| {
            let mut p = make_zero_poly(&mut ctx);
            ntt(&mut ctx, &mut p);
            p
        })
        .collect();

    // Message: alternating bits
    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    // Zero randomness
    let cbd_eta1 = (N as u32 * params.eta1 / 4) as usize;
    let cbd_eta2 = (N as u32 * params.eta2 / 4) as usize;
    let r_bytes: Vec<Vec<_>> = (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_eta1)).collect();
    let e1_bytes: Vec<Vec<_>> = (0..k).map(|_| make_zero_bytes(&mut ctx, cbd_eta2)).collect();
    let e2_bytes = make_zero_bytes(&mut ctx, cbd_eta2);

    // Encrypt + Decrypt
    let (u_vec, v) = encrypt(
        &mut ctx, params, &a_hat, &t_hat, &message_bits, &r_bytes, &e1_bytes, &e2_bytes,
    );
    let _recovered_bits = decrypt(&mut ctx, &s_hat, &u_vec, &v);

    let circuit_build_time = t_start.elapsed();
    eprintln!("Circuit build: {circuit_build_time:.2?}");
    eprintln!("Gate stats: {:?}", ctx.stats);

    // --- Finalize ---
    let t_finalize = Instant::now();
    ctx.finalize_guessed_vars();
    finalize_context(&mut ctx);
    ctx.validate_circuit();
    let finalize_time = t_finalize.elapsed();
    eprintln!("Finalize + validate: {finalize_time:.2?}");

    // --- Preprocess ---
    let t_preprocess = Instant::now();
    let preprocessed_circuit = PreprocessedCircuit::preprocess_circuit(&mut ctx);
    let preprocess_time = t_preprocess.elapsed();
    eprintln!("Preprocess: {preprocess_time:.2?}");
    eprintln!(
        "Trace log size: {}",
        preprocessed_circuit.params.trace_log_size
    );

    // --- Prove ---
    let pcs_config = PcsConfig::default();
    let t_prove = Instant::now();
    let circuit_proof = prove_circuit_assignment(
        ctx.values(),
        &preprocessed_circuit,
        &BaseColumnPool::<SimdBackend>::new(),
        pcs_config,
    );
    let prove_time = t_prove.elapsed();
    eprintln!("Prove: {prove_time:.2?}");

    assert!(
        circuit_proof.stark_proof.is_ok(),
        "Proof generation failed: {}",
        circuit_proof.stark_proof.err().unwrap()
    );

    // --- Proof size ---
    let preprocessed_column_ids = preprocessed_circuit.preprocessed_trace.ids();
    let proof_config = ProofConfig::from_components(
        &all_circuit_components::<QM31>(),
        preprocessed_column_ids.len(),
        &circuit_proof.pcs_config,
        INTERACTION_POW_BITS,
    );

    let (proof, _public_data) =
        prepare_circuit_proof_for_circuit_verifier(circuit_proof, &proof_config);

    let mut serialized = vec![];
    proof.serialize(&mut serialized);
    let proof_size_bytes = serialized.len();

    let total_time = t_start.elapsed();
    eprintln!("\n=== ML-KEM-512 Circuit Proof Summary ===");
    eprintln!("Total wall time:  {total_time:.2?}");
    eprintln!("  Circuit build:  {circuit_build_time:.2?}");
    eprintln!("  Finalize:       {finalize_time:.2?}");
    eprintln!("  Preprocess:     {preprocess_time:.2?}");
    eprintln!("  Prove:          {prove_time:.2?}");
    eprintln!("Proof size:       {} bytes ({:.1} KB)", proof_size_bytes, proof_size_bytes as f64 / 1024.0);
}
