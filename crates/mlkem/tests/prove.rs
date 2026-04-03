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
use mlkem::kem::{MLKEM_512, recipient_circuit, sender_circuit};
use mlkem::ntt::{RqPoly, ntt};
use mlkem::zq::{zq_witness, ZqVar};

fn make_zero_poly(ctx: &mut Context<QM31>) -> RqPoly {
    let coeffs: [ZqVar; 256] = std::array::from_fn(|_| zq_witness(ctx, 0));
    RqPoly { coeffs }
}

fn prove_and_measure(ctx: &mut Context<QM31>, label: &str) {
    let t0 = Instant::now();

    eprintln!("\n=== {label} ===");
    eprintln!("Gate stats: {:?}", ctx.stats);

    let t_fin = Instant::now();
    ctx.finalize_guessed_vars();
    finalize_context(ctx);
    ctx.validate_circuit();
    let finalize_time = t_fin.elapsed();

    let t_pre = Instant::now();
    let preprocessed = PreprocessedCircuit::preprocess_circuit(ctx);
    let preprocess_time = t_pre.elapsed();
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

    assert!(
        circuit_proof.stark_proof.is_ok(),
        "Proof failed: {}",
        circuit_proof.stark_proof.err().unwrap()
    );

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
    eprintln!("Proof size:  {} bytes ({:.1} KB)", serialized.len(), serialized.len() as f64 / 1024.0);
    eprintln!("Prove time:  {prove_time:.2?}");
    eprintln!("Total:       {total:.2?} (finalize {finalize_time:.2?} + preprocess {preprocess_time:.2?} + prove {prove_time:.2?})");
}

/// Benchmark: Sender circuit (Encaps) — derive shared key when sending.
#[test]
fn bench_sender() {
    let t_start = Instant::now();
    let mut ctx = Context::<QM31>::default();
    let params = &MLKEM_512;
    let k = params.k;

    // Public inputs: recipient's pk from PKI
    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    let t_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();

    // Witness: sender's random message
    let message_bits: Vec<_> = (0..N)
        .map(|i| guess(&mut ctx, QM31::from(M31::from((i % 2) as u32))))
        .collect();

    let _ss = sender_circuit(&mut ctx, params, &rho, &t_hat, &message_bits);

    eprintln!("Circuit build: {:.2?}", t_start.elapsed());
    prove_and_measure(&mut ctx, "ML-KEM-512 SENDER (Encaps + Blake2s)");
}

/// Benchmark: Recipient circuit (Decaps) — derive shared key when receiving.
#[test]
fn bench_recipient() {
    let t_start = Instant::now();
    let mut ctx = Context::<QM31>::default();
    let k = MLKEM_512.k;

    // Public inputs: own pk + received ciphertext
    let rho: Vec<_> = (0..2)
        .map(|i| guess(&mut ctx, QM31::from(M31::from(42u32 + i))))
        .collect();
    let u: Vec<RqPoly> = (0..k).map(|_| make_zero_poly(&mut ctx)).collect();
    let v = make_zero_poly(&mut ctx);

    // Witness: recipient's secret key
    let s_hat: Vec<RqPoly> = (0..k)
        .map(|_| { let mut p = make_zero_poly(&mut ctx); ntt(&mut ctx, &mut p); p })
        .collect();

    let _ss = recipient_circuit(&mut ctx, &rho, &s_hat, &u, &v);

    eprintln!("Circuit build: {:.2?}", t_start.elapsed());
    prove_and_measure(&mut ctx, "ML-KEM-512 RECIPIENT (Decaps + Blake2s)");
}
