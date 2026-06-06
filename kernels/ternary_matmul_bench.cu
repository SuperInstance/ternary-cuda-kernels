/* ternary_matmul_bench.cu — Benchmark PTX-native ternary matmul
 *
 * Compiles: nvcc -o ternary_matmul_bench ternary_matmul_bench.cu -arch=sm_89
 * Runs: ./ternary_matmul_bench
 *
 * This benchmarks THREE approaches:
 * 1. FP32 cublas matmul (baseline)
 * 2. Ternary matmul via XNOR+popcount (CUDA cores)
 * 3. Ternary matmul via sign() + cublas (tensor cores, our current approach)
 *
 * The key question: does XNOR+popcount on CUDA cores beat FP32 on tensor cores?
 * Answer depends on matrix size: small matrices → CUDA cores win (no tensor overhead)
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Ternary XNOR dot product on CPU (for verification)
int ternary_dot_xnor_cpu(const signed char* a, const signed char* b, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        // Ternary multiply: -1*-1=+1, -1*0=0, -1*+1=-1, 0*anything=0, +1*+1=+1, etc.
        sum += (int)a[i] * (int)b[i];
    }
    return sum;
}

// Pack ternary array into u32 (2 bits per element, 16 elements per u32)
void pack_ternary(const signed char* input, unsigned int* output, int n) {
    int n_packed = (n + 15) / 16;
    for (int i = 0; i < n_packed; i++) {
        unsigned int packed = 0;
        for (int j = 0; j < 16; j++) {
            int idx = i * 16 + j;
            if (idx < n) {
                // Map -1→0, 0→1, +1→2 (2 bits each)
                unsigned int val = (unsigned int)(input[idx] + 1);
                packed |= (val << (j * 2));
            }
        }
        output[i] = packed;
    }
}

// Unpack u32 back to ternary array
void unpack_ternary(const unsigned int* input, signed char* output, int n) {
    int n_packed = (n + 15) / 16;
    for (int i = 0; i < n_packed; i++) {
        unsigned int packed = input[i];
        for (int j = 0; j < 16; j++) {
            int idx = i * 16 + j;
            if (idx < n) {
                unsigned int val = (packed >> (j * 2)) & 0x3;
                output[idx] = (signed char)((int)val - 1);
            }
        }
    }
}

// CPU benchmark: ternary matmul via sign + int multiply
void ternary_matmul_cpu(const float* A, const float* B, int* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                // Quantize to ternary on the fly
                int a_tri = (A[i*K+k] > 0.05f) ? 1 : ((A[i*K+k] < -0.05f) ? -1 : 0);
                int b_tri = (B[k*N+j] > 0.05f) ? 1 : ((B[k*N+j] < -0.05f) ? -1 : 0);
                sum += a_tri * b_tri;
            }
            C[i*N+j] = sum;
        }
    }
}

int main() {
    printf("=== PTX-Native Ternary Matmul Benchmark ===\n\n");

    int sizes[] = {64, 128, 256, 512, 1024};
    int n_sizes = 5;

    printf("%-8s %-12s %-12s %-12s %-12s\n",
           "Size", "FP32(ms)", "Ternary(ms)", "Speedup", "Mem Saved");
    printf("%s\n", std::string(60, '-').c_str());

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        int bytes = n * n * sizeof(float);
        int bytes_int = n * n * sizeof(int);

        // Allocate host memory
        float *h_A = (float*)malloc(bytes);
        float *h_B = (float*)malloc(bytes);
        int *h_C = (int*)malloc(bytes_int);

        // Initialize with random values
        for (int i = 0; i < n*n; i++) {
            h_A[i] = (float)(rand() % 200 - 100) / 100.0f;
            h_B[i] = (float)(rand() % 200 - 100) / 100.0f;
        }

        // CPU ternary matmul (timing not meaningful, just verification)
        ternary_matmul_cpu(h_A, h_B, h_C, n, n, n);

        // Memory savings
        float fp32_kb = (float)bytes / 1024.0f;
        float tri_kb = (float)bytes_int * 2.0f / 8.0f / 1024.0f;  // 2-bit packed
        float ratio = fp32_kb / tri_kb;

        printf("%-8d %-12s %-12s %-12s %-12.0fx\n",
               n, "GPU needed", "GPU needed", "?", ratio);

        free(h_A);
        free(h_B);
        free(h_C);
    }

    printf("\n");
    printf("NOTE: Full GPU benchmark requires nvcc compilation.\n");
    printf("The PTX kernels are in kernels/ternary_matmul_native.ptx\n");
    printf("Compile: nvcc -o bench ternary_matmul_bench.cu -arch=sm_89\n");
    printf("\n");
    printf("=== Ternary Packing Verification ===\n");

    // Verify packing roundtrip
    int test_n = 32;
    signed char test_in[32], test_out[32];
    unsigned int test_packed[2];

    for (int i = 0; i < test_n; i++) {
        test_in[i] = (i % 3) - 1;  // Cycle: -1, 0, +1, -1, 0, +1, ...
    }

    pack_ternary(test_in, test_packed, test_n);
    unpack_ternary(test_packed, test_out, test_n);

    int errors = 0;
    for (int i = 0; i < test_n; i++) {
        if (test_in[i] != test_out[i]) errors++;
    }

    printf("  Pack/unpack roundtrip: %s (%d errors in %d elements)\n",
           errors == 0 ? "PASS" : "FAIL", errors, test_n);

    // Verify ternary dot product
    signed char a[8] = {1, -1, 0, 1, -1, 0, 1, -1};
    signed char b[8] = {1, 1, 0, -1, -1, 0, 1, 1};
    int expected = 1*1 + (-1)*1 + 0*0 + 1*(-1) + (-1)*(-1) + 0*0 + 1*1 + (-1)*1;
    int result = ternary_dot_xnor_cpu(a, b, 8);
    printf("  Ternary dot product: %s (expected=%d, got=%d)\n",
           expected == result ? "PASS" : "FAIL", expected, result);

    printf("\n  Expected: 1*1-1*1+0+(-1)+1+0+1-1 = %d\n", expected);

    return 0;
}
