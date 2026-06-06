# ternary-cuda-kernels

GPU-accelerated ternary operations compiled to PTX — jam session simulation, ternary matrix multiply via XNOR+popcount, and harmony reduction, all with CPU reference implementations for verification.

## Why This Exists

When you're running 10,000 jam sessions in parallel, each with 4+ voices over 100+ ticks, you need GPU acceleration. But CUDA kernels for ternary operations (−1, 0, +1) aren't standard. This crate provides three compiled PTX kernels plus their CPU reference implementations, so you can verify correctness on CPU and run at scale on GPU.

The key optimization: **2-bit trit packing**. Three trit values (−1, 0, +1) map to 2 bits each, so 16 trits pack into a single `u32`. Ternary matrix multiply then becomes XNOR+popcount — 16× denser than floating point, with bitwise operations instead of multiply-accumulate.

## Architecture

```text
PTX Kernels (compiled from kernels/ternary_jam.cu)
├── jam_session_kernel     — 10,000 parallel jam sessions
├── ternary_matmul_kernel  — XNOR+popcount matrix multiply
└── harmony_reduce_kernel  — Reduce harmony scores across sessions

CPU Reference (in Rust)
├── jam_session_cpu()      — Identical logic, single-threaded
├── compute_harmony()      — Pairwise consonance/dissonance
├── apply_rule()           — Four improv rules (Free/Parallel/Contrary/Resolve)
├── mix_voices()           — Majority vote across voices
└── Trit packing/unpacking — 2-bit encode/decode
```

### Z₃ Arithmetic

The ternary operations use Z₃ (the cyclic group of order 3):

| a | b | tadd(a,b) | tmul(a,b) |
|---|---|-----------|-----------|
| −1 | −1 | +1 | +1 |
| −1 | 0 | −1 | 0 |
| −1 | +1 | 0 | −1 |
| 0 | 0 | 0 | 0 |
| +1 | +1 | −1 | +1 |

Z₃ addition wraps: (+1) + (+1) = −1, (−1) + (−1) = +1. Z₃ multiplication is standard sign multiplication.

### Jam Session Model

Each voice follows an improv rule applied to its previous state and tendency:

| Rule | Name | Operation |
|------|------|-----------|
| 0 | Free | Output = tendency |
| 1 | Parallel | Output = tadd(prev, tendency) |
| 2 | Contrary | Output = tadd(prev, −tendency) |
| 3 | Resolve | Drift toward 0 |

Harmony is computed pairwise: matching non-zero voices = consonance, clashing non-zero voices = dissonance.

## Usage

```rust
use ternary_cuda_kernels::*;

// Trit packing — 16 trits in one u32
let vals: Vec<i8> = (0..16).map(|i| [1, -1, 0][i % 3]).collect();
let packed = pack_16_trits(&vals);
let unpacked = unpack_16_trits(packed);
assert_eq!(&unpacked[..], &vals[..]);

// Z₃ arithmetic
assert_eq!(tadd(1, 1), -1);   // wraps in Z₃
assert_eq!(tadd(-1, 1), 0);   // cancellation
assert_eq!(tmul(-1, -1), 1);  // double negative

// CPU jam session reference
let tendencies = [1i8, -1, 1, -1];
let rules = [0u8, 2, 1, 3]; // Free, Contrary, Parallel, Resolve
let (output, consonance, dissonance) = jam_session_cpu(4, 100, &tendencies, &rules);
assert_eq!(output.len(), 100);
for &v in &output { assert!(v >= -1 && v <= 1); }

// Load PTX for GPU execution
let kernels = load_ptx();
assert_eq!(kernels.len(), 3);
for k in &kernels {
    assert!(!k.ptx.is_empty());
}

// Harmony computation
let (c, d) = compute_harmony(&[1, 1, 1]);
assert_eq!(c, 3); assert_eq!(d, 0); // All agree = consonant

let (c, d) = compute_harmony(&[1, -1]);
assert_eq!(c, 0); assert_eq!(d, 1); // Clash = dissonant
```

## API Reference

### Trit Packing
- `pack_trit(v: i8)` → `u8` — 2-bit encode (−1→00, 0→01, +1→10)
- `unpack_trit(bits: u8)` → `i8` — 2-bit decode
- `pack_16_trits(values: &[i8])` → `u32` — Pack 16 trits into one word
- `unpack_16_trits(packed: u32)` → `[i8; 16]` — Unpack one word to 16 trits

### Z₃ Arithmetic
- `tadd(a: i8, b: i8)` → `i8` — Z₃ addition (explicit 9-case match)
- `tmul(a: i8, b: i8)` → `i8` — Z₃ multiplication (sign product)

### Harmony
- `harmony_score(consonance, dissonance)` → `i64` — `consonance - dissonance`
- `compute_harmony(voices: &[i8])` → `(consonance, dissonance)` — Pairwise comparison

### Jam Session
- `apply_rule(prev, tendency, rule)` → `i8` — Apply one of 4 improv rules
- `mix_voices(voices: &[i8])` → `i8` — Majority vote across voices
- `jam_session_cpu(n_voices, n_ticks, tendencies, rules)` → `(output, consonance, dissonance)`

### PTX Loading
- `PtxKernel { name, ptx }` — Kernel name + compiled PTX bytes
- `load_ptx()` → `Vec<PtxKernel>` — Load all 3 kernels

## The Deeper Idea

This crate is the bridge between musical cognition models (musician-soul) and GPU hardware. The jam session model is deliberately simple — ternary voices with four improv rules — because simplicity enables massive parallelism. Each jam session is independent, so 10,000 sessions map cleanly to 10,000 GPU threads.

The 2-bit trit packing isn't just compression — it enables XNOR+popcount matrix multiply, which is the standard trick for binary neural networks extended to ternary. Three states instead of two gives you zero as a "don't care" value, which is useful for sparse activation patterns.

## Related Crates

- [`ternary-cuda-kernels-v2`](../ternary-cuda-kernels-v2) — Adds groove scheduling, voice leading, and harmony remap kernels
- [`musician-soul`](../musician-soul) — The persona system that generates the jam sessions these kernels accelerate
- [`ternary-auto-vectorizer`](../ternary-auto-vectorizer) — Formal verification that scalar and vectorized ternary ops are equivalent
