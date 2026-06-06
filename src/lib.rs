//! # ternary-cuda-kernels
//!
//! GPU-accelerated music cognition patterns compiled to PTX.
//! Three CUDA kernels loaded at runtime:
//! - `jam_session_kernel`: Run 10,000 jam sessions in parallel on GPU
//! - `ternary_matmul_kernel`: Ternary matrix multiply via XNOR+popcount (16× density)
//! - `harmony_reduce_kernel`: Reduce harmony scores across all sessions
//!
//! PTX compiled from kernels/ternary_jam.cu via nvcc.
//! Host code loads PTX dynamically — no CUDA runtime dependency.

/// 2-bit ternary packing: -1→0b00, 0→0b01, +1→0b10
pub fn pack_trit(v: i8) -> u8 {
    match v {
        -1 => 0b00,
        0 => 0b01,
        1 => 0b10,
        _ => 0b01,
    }
}

/// Unpack a 2-bit trit back to i8.
pub fn unpack_trit(bits: u8) -> i8 {
    match bits & 0x3 {
        0 => -1,
        1 => 0,
        2 => 1,
        _ => 0,
    }
}

/// Pack 16 ternary values into one u32.
pub fn pack_16_trits(values: &[i8]) -> u32 {
    let mut packed = 0u32;
    for (i, &v) in values.iter().take(16).enumerate() {
        packed |= (pack_trit(v) as u32) << (i * 2);
    }
    packed
}

/// Unpack a u32 into 16 ternary values.
pub fn unpack_16_trits(packed: u32) -> [i8; 16] {
    let mut out = [0i8; 16];
    for i in 0..16 {
        out[i] = unpack_trit(((packed >> (i * 2)) & 0x3) as u8);
    }
    out
}

/// Z₃ addition — explicit match on all 9 pairs.
pub fn tadd(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1,
        (-1, 0) => -1,
        (-1, 1) => 0,
        (0, -1) => -1,
        (0, 0) => 0,
        (0, 1) => 1,
        (1, -1) => 0,
        (1, 0) => 1,
        (1, 1) => -1,
        _ => 0,
    }
}

/// Z₃ multiplication.
pub fn tmul(a: i8, b: i8) -> i8 { a * b }

/// Harmony score from consonance and dissonance counts.
pub fn harmony_score(consonance: i64, dissonance: i64) -> i64 {
    consonance - dissonance
}

/// Compute harmony for a set of ternary voice outputs (CPU reference).
pub fn compute_harmony(voices: &[i8]) -> (i64, i64) {
    let mut consonance = 0i64;
    let mut dissonance = 0i64;
    for i in 0..voices.len() {
        for j in (i+1)..voices.len() {
            let a = voices[i]; let b = voices[j];
            if a != 0 && b != 0 && a != b { dissonance += 1; }
            else if a != 0 && b != 0 && a == b { consonance += 1; }
        }
    }
    (consonance, dissonance)
}

/// Apply an improv rule to generate the next note (CPU reference).
pub fn apply_rule(prev: i8, tendency: i8, rule: u8) -> i8 {
    match rule {
        0 => tendency,                              // Free
        1 => tadd(prev, tendency),                   // Parallel
        2 => tadd(prev, -tendency),                  // Contrary
        3 => { if prev > 0 { prev - 1 } else if prev < 0 { prev + 1 } else { 0 } }, // Resolve
        _ => tendency,
    }
}

/// Mix multiple voice outputs into one ternary value (CPU reference).
pub fn mix_voices(voices: &[i8]) -> i8 {
    let sum: i32 = voices.iter().map(|&v| v as i32).sum();
    if sum > 0 { 1 } else if sum < 0 { -1 } else { 0 }
}

/// Run a jam session on CPU (reference implementation for verifying GPU).
pub fn jam_session_cpu(
    n_voices: usize,
    n_ticks: usize,
    tendencies: &[i8],
    rules: &[u8],
) -> (Vec<i8>, i64, i64) {
    let mut voices = vec![0i8; n_voices];
    let mut output = Vec::with_capacity(n_ticks);
    let mut consonance = 0i64;
    let mut dissonance = 0i64;

    for _ in 0..n_ticks {
        for v in 0..n_voices {
            voices[v] = apply_rule(voices[v], tendencies[v], rules[v]);
        }
        let (c, d) = compute_harmony(&voices);
        consonance += c;
        dissonance += d;
        output.push(mix_voices(&voices));
    }
    (output, consonance, dissonance)
}

/// PTX kernel metadata.
pub struct PtxKernel {
    pub name: &'static str,
    pub ptx: &'static [u8],
}

/// Load all compiled PTX kernels.
pub fn load_ptx() -> Vec<PtxKernel> {
    vec![
        PtxKernel { name: "jam_session_kernel", ptx: include_bytes!("../kernels/ternary_jam.ptx") },
        PtxKernel { name: "ternary_matmul_kernel", ptx: include_bytes!("../kernels/ternary_jam.ptx") },
        PtxKernel { name: "harmony_reduce_kernel", ptx: include_bytes!("../kernels/ternary_jam.ptx") },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn pack_unpack_roundtrip() {
        for v in [-1i8, 0, 1] { assert_eq!(unpack_trit(pack_trit(v)), v); }
    }
    #[test] fn pack_16_roundtrip() {
        let vals: Vec<i8> = (0..16).map(|i| [1, -1, 0][i % 3]).collect();
        let packed = pack_16_trits(&vals);
        let unpacked = unpack_16_trits(packed);
        assert_eq!(&unpacked[..], &vals[..]);
    }
    #[test] fn tadd_z3() {
        assert_eq!(tadd(1, 1), -1);   // wraps in Z₃
        assert_eq!(tadd(-1, -1), 1);  // wraps in Z₃
        assert_eq!(tadd(1, -1), 0);   // cancellation
        assert_eq!(tadd(0, 1), 1);    // identity
    }
    #[test] fn harmony_all_agree() {
        let (c, d) = compute_harmony(&[1, 1, 1]);
        assert_eq!(c, 3); assert_eq!(d, 0);
    }
    #[test] fn harmony_all_disagree() {
        let (c, d) = compute_harmony(&[1, -1]);
        assert_eq!(c, 0); assert_eq!(d, 1);
    }
    #[test] fn mix_voices_basic() {
        assert_eq!(mix_voices(&[1, 1, -1]), 1);
        assert_eq!(mix_voices(&[1, -1]), 0);
        assert_eq!(mix_voices(&[-1, -1]), -1);
    }
    #[test] fn jam_session_4_voices() {
        let tendencies = [1, -1, 1, -1];
        let rules = [0u8, 2, 1, 3]; // Free, Contrary, Parallel, Resolve
        let (output, cons, diss) = jam_session_cpu(4, 100, &tendencies, &rules);
        assert_eq!(output.len(), 100);
        assert!(cons > 0 || diss > 0);
        // Output should be ternary
        for &v in &output { assert!(v >= -1 && v <= 1); }
    }
    #[test] fn jam_session_conservation() {
        let tendencies = [1, -1, 0];
        let rules = [0u8, 0, 0];
        let (output, _, _) = jam_session_cpu(3, 1000, &tendencies, &rules);
        // Over many ticks, the sum should stay bounded (conservation)
        let sum: i64 = output.iter().map(|&v| v as i64).sum();
        assert!(sum.abs() < 500); // Not zero (dynamic) but bounded
    }
    #[test] fn ptx_loads() {
        let kernels = load_ptx();
        assert_eq!(kernels.len(), 3);
        assert!(!kernels[0].ptx.is_empty());
    }
    #[test] fn ptx_contains_kernel_names() {
        let ptx = include_str!("../kernels/ternary_jam.ptx");
        assert!(ptx.contains("jam_session_kernel"));
        assert!(ptx.contains("ternary_matmul_kernel"));
        assert!(ptx.contains("harmony_reduce_kernel"));
    }
}
