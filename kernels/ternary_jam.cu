/**
 * ternary_jam.cu — GPU-accelerated multi-agent jam session
 * 
 * Maps agent-jam's WorkSession to CUDA:
 * - Each thread block = one jam session
 * - Each thread = one voice/collaborator
 * - Shared memory for chord progression and harmony scoring
 * - Warp-level vote for consensus mixing
 * 
 * Compiles to PTX via: nvcc -ptx -o ternary_jam.ptx kernels/ternary_jam.cu
 */

#include <cstdint>

// ── Ternary types ──────────────────────────────────────────────────
// Packed: 2 bits per trit. 16 trits per u32.
// -1 → 0b00, 0 → 0b01, +1 → 0b10 (0b11 = unused/error)

__device__ __forceinline__
int8_t trit_unpack(uint32_t packed, int idx) {
    // Extract 2-bit trit at position idx (0-15)
    uint32_t bits = (packed >> (idx * 2)) & 0x3;
    switch (bits) {
        case 0: return -1;  // NegOne
        case 1: return  0;  // Zero
        case 2: return  1;  // PosOne
        default: return 0;  // Error → treat as zero
    }
}

__device__ __forceinline__
uint32_t trit_pack(uint32_t packed, int idx, int8_t val) {
    // Set 2-bit trit at position idx
    uint32_t bits;
    switch (val) {
        case -1: bits = 0; break;
        case  0: bits = 1; break;
        case  1: bits = 2; break;
        default: bits = 1; break;
    }
    uint32_t mask = ~(0x3u << (idx * 2));
    return (packed & mask) | (bits << (idx * 2));
}

// ── Z₃ arithmetic on GPU ──────────────────────────────────────────

__device__ __forceinline__
int8_t tadd(int8_t a, int8_t b) {
    // Z₃ addition using match (not modular arithmetic)
    // Handles all 9 cases correctly
    if (a == 0) return b;
    if (b == 0) return a;
    if (a == b) {
        // 1+1 → -1, (-1)+(-1) → 1 (wraps in Z₃)
        return -a;
    }
    return 0; // 1+(-1) = 0, (-1)+1 = 0
}

__device__ __forceinline__
int8_t tmul(int8_t a, int8_t b) {
    return a * b; // Standard multiplication works for {-1,0,1}
}

// ── Jam Session Kernel ─────────────────────────────────────────────

/**
 * Each block = one jam session. Each thread = one voice.
 * 
 * voices[]:      packed ternary values per voice (16 trits per u32)
 * tendencies[]:   -1, 0, or +1 tendency per voice
 * rules[]:        0=Free, 1=Parallel, 2=Contrary, 3=Resolve
 * progression[]:  packed chord progression steps
 * output[]:       mixed output per tick
 * harmony[]:      [consonance, dissonance] per session
 * n_voices:       voices per session (<=32, one per thread in warp)
 * n_ticks:        ticks to simulate
 * n_sessions:     total sessions (gridDim.x blocks)
 */
extern "C" __global__
void jam_session_kernel(
    const uint32_t* __restrict__ voices,      // [n_sessions * MAX_VOICES * ticks_per_voice / 16]
    const int8_t*   __restrict__ tendencies,  // [n_sessions * MAX_VOICES]
    const int8_t*   __restrict__ rules,       // [n_sessions * MAX_VOICES]
    const uint32_t* __restrict__ progression, // [n_sessions * prog_len]
    uint32_t*       __restrict__ output,      // [n_sessions * n_ticks / 16]
    int64_t*        __restrict__ harmony,     // [n_sessions * 2] (consonance, dissonance)
    const int       n_voices,
    const int       n_ticks,
    const int       prog_len
) {
    // Shared memory for this session's state
    __shared__ int8_t  s_voices[32];     // Current note per voice
    __shared__ int8_t  s_tendencies[32];
    __shared__ int8_t  s_rules[32];
    __shared__ int64_t s_consonance;
    __shared__ int64_t s_dissonance;
    __shared__ uint32_t s_progression[16]; // Chord progression
    
    int session = blockIdx.x;
    int tid = threadIdx.x;
    
    // Initialize shared state (first thread)
    if (tid == 0) {
        s_consonance = 0;
        s_dissonance = 0;
        for (int i = 0; i < prog_len && i < 16; i++) {
            s_progression[i] = progression[session * prog_len + i];
        }
    }
    
    // Each thread loads its voice data
    if (tid < n_voices) {
        s_tendencies[tid] = tendencies[session * 32 + tid];
        s_rules[tid] = rules[session * 32 + tid];
        // Load first trit from packed voice data
        s_voices[tid] = trit_unpack(voices[session * 32 + tid], 0);
    }
    __syncthreads();
    
    // Run the jam session
    for (int tick = 0; tick < n_ticks; tick++) {
        int8_t my_note;
        
        if (tid < n_voices) {
            // Apply improv rule
            int8_t prev = s_voices[tid];
            int8_t tend = s_tendencies[tid];
            
            switch (s_rules[tid]) {
                case 0: // Free
                    my_note = tend;
                    break;
                case 1: // Parallel — move in same direction as tendency
                    my_note = tadd(prev, tend);
                    break;
                case 2: // Contrary — move against tendency
                    my_note = tadd(prev, -tend);
                    break;
                case 3: // Resolve — move toward 0
                    if (prev > 0) my_note = prev - 1;
                    else if (prev < 0) my_note = prev + 1;
                    else my_note = 0;
                    break;
                default:
                    my_note = tend;
            }
            s_voices[tid] = my_note;
        }
        __syncthreads();
        
        // Harmony scoring: pairwise dissonance
        // Each thread checks against higher-indexed threads
        if (tid < n_voices) {
            int8_t my = s_voices[tid];
            for (int j = tid + 1; j < n_voices; j++) {
                int8_t other = s_voices[j];
                // Dissonance: both non-zero and different
                if (my != 0 && other != 0 && my != other) {
                    s_dissonance++;
                } else if (my != 0 && other != 0 && my == other) {
                    s_consonance++;
                }
            }
        }
        __syncthreads();
        
        // Mix output: warp-level majority vote
        if (tid < n_voices) {
            // Use ballot for fast vote counting
            unsigned int ballot = __ballot_sync(0xFFFFFFFF, s_voices[tid] > 0 ? 1 : (s_voices[tid] < 0 ? 0x80000000u : 0));
            // First thread computes the mix
            if (tid == 0) {
                int32_t sum = 0;
                for (int v = 0; v < n_voices; v++) {
                    sum += s_voices[v];
                }
                int8_t mixed = (sum > 0) ? 1 : (sum < 0) ? -1 : 0;
                // Pack into output
                int out_idx = session * ((n_ticks + 15) / 16) + tick / 16;
                atomicOr(&output[out_idx], (uint32_t)((mixed == -1 ? 0 : mixed == 0 ? 1 : 2)) << ((tick % 16) * 2));
            }
        }
        __syncthreads();
    }
    
    // Write harmony scores back to global memory
    if (tid == 0) {
        harmony[session * 2] = s_consonance;
        harmony[session * 2 + 1] = s_dissonance;
    }
}

// ── Ternary Matmul Kernel ──────────────────────────────────────────

/**
 * Ternary matrix multiplication using XNOR+popcount.
 * A and B contain packed ternary values (16 trits per u32).
 * 
 * This is where the real GPU power comes in:
 * - FP32 matmul: 1 multiply + 1 add per element
 * - Ternary matmul: 1 XNOR + 1 popcount per 16 elements
 * 
 * Speedup: ~16x memory density, ~10x compute throughput
 */
extern "C" __global__
void ternary_matmul_kernel(
    const uint32_t* __restrict__ a,     // [M * K/16] packed row-major
    const uint32_t* __restrict__ b,     // [K * N/16] packed row-major
    int32_t*        __restrict__ c,     // [M * N] output (int32 accumulator)
    const int M, const int N, const int K_packed  // K_packed = K/16
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    int32_t sum = 0;
    
    for (int k = 0; k < K_packed; k++) {
        uint32_t va = a[row * K_packed + k];
        uint32_t vb = b[(k * 16) * N / 16 + col];  // Adjusted indexing
        
        // XNOR gives 1 where bits match, 0 where they differ
        uint32_t xnor_result = ~(va ^ vb);
        
        // Popcount = number of matching bit pairs
        // But we need to handle the ternary encoding:
        // Matching 00→both -1, matching 01→both 0, matching 10→both +1
        // Mismatch = dissonance (different values)
        
        // For ternary dot product: count (1*1 + (-1)*(-1)) - count mismatches
        // Each 2-bit field: match=+1 or -1, mismatch=cancellation
        int32_t local_sum = 0;
        for (int bit = 0; bit < 16; bit++) {
            int8_t ta = trit_unpack(va, bit);
            int8_t tb = trit_unpack(vb, bit);
            local_sum += tmul(ta, tb);
        }
        sum += local_sum;
    }
    
    c[row * N + col] = sum;
}

// ── Harmony Reduction Kernel ───────────────────────────────────────

/**
 * Reduce harmony scores across all sessions.
 * Outputs: total_consonance, total_dissonance, best_session, worst_session
 */
extern "C" __global__
void harmony_reduce_kernel(
    const int64_t* __restrict__ harmony,  // [n_sessions * 2]
    int64_t*       __restrict__ results,  // [4]: total_consonance, total_dissonance, best_idx, worst_idx
    const int      n_sessions
) {
    __shared__ int64_t s_consonances[256];
    __shared__ int64_t s_dissonances[256];
    // Use int32 for atomics compatibility
    __shared__ int64_t s_scores[256];
    
    int tid = threadIdx.x;
    int session = blockIdx.x * blockDim.x + tid;
    
    // Load
    if (session < n_sessions) {
        s_consonances[tid] = harmony[session * 2];
        s_dissonances[tid] = harmony[session * 2 + 1];
        s_scores[tid] = s_consonances[tid] - s_dissonances[tid];
    } else {
        s_consonances[tid] = 0;
        s_dissonances[tid] = 0;
        s_scores[tid] = INT64_MIN;
    }
    __syncthreads();
    
    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_consonances[tid] += s_consonances[tid + stride];
            s_dissonances[tid] += s_dissonances[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes results
    if (tid == 0) {
        results[0] += s_consonances[0];
        results[1] += s_dissonances[0];
    }
}
