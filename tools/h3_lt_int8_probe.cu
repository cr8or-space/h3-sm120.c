/* Which INT8 matmul variants cuBLASLt accepts on this GPU, and whether the ones
 * it accepts compute the right thing.
 *
 * The DiT's INT8 linear is D = A^T B with A the K x M int8 weight (one column
 * per output channel), B the K x N int8 activations (one column per token) and
 * D the M x N result. Today it writes an int32 accumulator and a separate
 * kernel applies weight_scales[i] * input_scales[j]. This probe asks whether
 * that rescale can move into the GEMM: a float D, an outer-vector scale, an
 * alpha vector, or one of the block scale modes.
 *
 * Each variant is checked against a host reference on a small random problem,
 * then timed at the DiT's QKV, FC1 and FC2 shapes on random operands, with
 * every algorithm the heuristic returns. */

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                  \
                    cudaGetErrorString(status));                               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

enum scaling {
    SCALE_NONE,
    SCALE_OUTER_VEC,   /* A scale per M, B scale per N */
    SCALE_SCALAR,      /* one float for A, one for B */
    SCALE_VEC128,      /* one float per 128 elements of K */
    SCALE_ALPHA_VECTOR /* alpha per row of D, beta zero */
};

typedef struct {
    const char *name;
    cudaDataType output;
    cublasComputeType_t compute;
    cudaDataType scale_type;
    enum scaling scaling;
} variant;

static const char *status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "ok";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
        case CUBLAS_STATUS_INVALID_VALUE: return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "NOT_SUPPORTED";
        default: return "other";
    }
}

static size_t element_bytes(cudaDataType type) {
    switch (type) {
        case CUDA_R_32F:
        case CUDA_R_32I: return 4;
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        default: return 1;
    }
}

/* BF16 is the top 16 bits of an IEEE float32; F16 is IEEE binary16. */
static float half_to_float(uint16_t bits, int bfloat) {
    if (bfloat) {
        uint32_t word = (uint32_t)bits << 16;
        float value;
        memcpy(&value, &word, sizeof(value));
        return value;
    }
    int sign = bits >> 15, exponent = (bits >> 10) & 0x1f;
    int mantissa = bits & 0x3ff;
    float value = exponent == 0 ? ldexpf((float)mantissa, -24)
                                : ldexpf((float)(mantissa | 0x400),
                                         exponent - 25);
    return sign ? -value : value;
}

static float read_output(const void *d, size_t index, cudaDataType type) {
    switch (type) {
        case CUDA_R_32F: return ((const float *)d)[index];
        case CUDA_R_32I: return (float)((const int32_t *)d)[index];
        case CUDA_R_16F:
            return half_to_float(((const uint16_t *)d)[index], 0);
        case CUDA_R_16BF:
            return half_to_float(((const uint16_t *)d)[index], 1);
        case CUDA_R_8I: return (float)((const int8_t *)d)[index];
        default: return NAN;
    }
}

typedef struct {
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a, b, d;
    cublasLtMatmulPreference_t preference;
    void *a_scale, *b_scale, *alpha_vector;
    float alpha_f, beta_f;
    int32_t alpha_i, beta_i;
} plan;

static void plan_destroy(plan *p) {
    if (p->preference) cublasLtMatmulPreferenceDestroy(p->preference);
    if (p->a) cublasLtMatrixLayoutDestroy(p->a);
    if (p->b) cublasLtMatrixLayoutDestroy(p->b);
    if (p->d) cublasLtMatrixLayoutDestroy(p->d);
    if (p->operation) cublasLtMatmulDescDestroy(p->operation);
    cudaFree(p->a_scale);
    cudaFree(p->b_scale);
    cudaFree(p->alpha_vector);
    memset(p, 0, sizeof(*p));
}

/* Builds the descriptors, with scale buffers filled from the given host
 * vectors (per-M and per-N; the block modes take their first element). */
static cublasStatus_t plan_create(plan *p, const variant *v, int m, int n,
                                  int k, const float *m_scales,
                                  const float *n_scales, size_t workspace) {
    memset(p, 0, sizeof(*p));
    cublasStatus_t status =
        cublasLtMatmulDescCreate(&p->operation, v->compute, v->scale_type);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    cublasOperation_t transpose = CUBLAS_OP_T, plain = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &transpose, sizeof(transpose));
    cublasLtMatmulDescSetAttribute(p->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &plain, sizeof(plain));
    if (v->scaling == SCALE_OUTER_VEC || v->scaling == SCALE_SCALAR ||
        v->scaling == SCALE_VEC128) {
        cublasLtMatmulMatrixScale_t mode =
            v->scaling == SCALE_OUTER_VEC ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
            : v->scaling == SCALE_SCALAR  ? CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F
                                          : CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
        size_t a_count = v->scaling == SCALE_OUTER_VEC ? (size_t)m
                         : v->scaling == SCALE_SCALAR
                             ? 1
                             : (size_t)m * ((k + 127) / 128);
        size_t b_count = v->scaling == SCALE_OUTER_VEC ? (size_t)n
                         : v->scaling == SCALE_SCALAR
                             ? 1
                             : (size_t)n * ((k + 127) / 128);
        float *a_host = (float *)malloc(a_count * sizeof(float));
        float *b_host = (float *)malloc(b_count * sizeof(float));
        for (size_t i = 0; i < a_count; i++)
            a_host[i] = v->scaling == SCALE_OUTER_VEC ? m_scales[i]
                                                      : m_scales[0];
        for (size_t i = 0; i < b_count; i++)
            b_host[i] = v->scaling == SCALE_OUTER_VEC ? n_scales[i]
                                                      : n_scales[0];
        CHECK(cudaMalloc(&p->a_scale, a_count * sizeof(float)));
        CHECK(cudaMalloc(&p->b_scale, b_count * sizeof(float)));
        CHECK(cudaMemcpy(p->a_scale, a_host, a_count * sizeof(float),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(p->b_scale, b_host, b_count * sizeof(float),
                         cudaMemcpyHostToDevice));
        free(a_host);
        free(b_host);
        if ((status = cublasLtMatmulDescSetAttribute(
                 p->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode,
                 sizeof(mode))) != CUBLAS_STATUS_SUCCESS ||
            (status = cublasLtMatmulDescSetAttribute(
                 p->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode,
                 sizeof(mode))) != CUBLAS_STATUS_SUCCESS)
            return status;
        cublasLtMatmulDescSetAttribute(p->operation,
                                       CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                       &p->a_scale, sizeof(p->a_scale));
        cublasLtMatmulDescSetAttribute(p->operation,
                                       CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                       &p->b_scale, sizeof(p->b_scale));
    }
    if (v->scaling == SCALE_ALPHA_VECTOR) {
        cublasLtPointerMode_t mode =
            CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
        if ((status = cublasLtMatmulDescSetAttribute(
                 p->operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode,
                 sizeof(mode))) != CUBLAS_STATUS_SUCCESS)
            return status;
        size_t bytes = (size_t)m * element_bytes(v->scale_type);
        CHECK(cudaMalloc(&p->alpha_vector, bytes));
        if (v->scale_type == CUDA_R_32I) {
            int32_t *host = (int32_t *)malloc(bytes);
            for (int i = 0; i < m; i++) host[i] = 1;
            CHECK(cudaMemcpy(p->alpha_vector, host, bytes,
                             cudaMemcpyHostToDevice));
            free(host);
        } else {
            CHECK(cudaMemcpy(p->alpha_vector, m_scales, bytes,
                             cudaMemcpyHostToDevice));
        }
    }
    p->alpha_f = 1.0f;
    p->alpha_i = 1;
    if ((status = cublasLtMatrixLayoutCreate(&p->a, CUDA_R_8I, k, m, k)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&p->b, CUDA_R_8I, k, n, k)) !=
            CUBLAS_STATUS_SUCCESS ||
        (status = cublasLtMatrixLayoutCreate(&p->d, v->output, m, n, m)) !=
            CUBLAS_STATUS_SUCCESS)
        return status;
    cublasLtMatmulPreferenceCreate(&p->preference);
    cublasLtMatmulPreferenceSetAttribute(
        p->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
        sizeof(workspace));
    return CUBLAS_STATUS_SUCCESS;
}

static const void *alpha_of(const plan *p, const variant *v) {
    if (v->scaling == SCALE_ALPHA_VECTOR) return p->alpha_vector;
    return v->scale_type == CUDA_R_32I ? (const void *)&p->alpha_i
                                       : (const void *)&p->alpha_f;
}

static const void *beta_of(const plan *p, const variant *v) {
    return v->scale_type == CUDA_R_32I ? (const void *)&p->beta_i
                                       : (const void *)&p->beta_f;
}

static void *upload(const void *host, size_t bytes) {
    void *device = NULL;
    CHECK(cudaMalloc(&device, bytes ? bytes : 1));
    CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    return device;
}

int main(void) {
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, 0));
    printf("device %s sm_%d%d, cuBLASLt %zu\n", properties.name,
           properties.major, properties.minor, cublasLtGetVersion());

    cublasLtHandle_t lt = NULL;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) return 1;
    size_t workspace_bytes = (size_t)128 << 20;
    void *workspace = NULL;
    CHECK(cudaMalloc(&workspace, workspace_bytes));

    const variant variants[] = {
        {"i32 C32I (today)", CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I,
         SCALE_NONE},
        {"i8 C32I", CUDA_R_8I, CUBLAS_COMPUTE_32I, CUDA_R_32F, SCALE_NONE},
        {"f32 C32I", CUDA_R_32F, CUBLAS_COMPUTE_32I, CUDA_R_32F, SCALE_NONE},
        {"bf16 C32I", CUDA_R_16BF, CUBLAS_COMPUTE_32I, CUDA_R_32F, SCALE_NONE},
        {"f16 C32I", CUDA_R_16F, CUBLAS_COMPUTE_32I, CUDA_R_32F, SCALE_NONE},
        {"f32 C32F", CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, SCALE_NONE},
        {"bf16 C32F", CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F, SCALE_NONE},
        {"i32 C32I alpha vec i32", CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32I,
         SCALE_ALPHA_VECTOR},
        {"i32 C32I alpha vec f32", CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32F,
         SCALE_ALPHA_VECTOR},
        {"f32 C32F alpha vec", CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         SCALE_ALPHA_VECTOR},
        {"bf16 C32F alpha vec", CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         SCALE_ALPHA_VECTOR},
        {"bf16 C32I outer vec", CUDA_R_16BF, CUBLAS_COMPUTE_32I, CUDA_R_32F,
         SCALE_OUTER_VEC},
        {"bf16 C32F outer vec", CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         SCALE_OUTER_VEC},
        {"f32 C32F outer vec", CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         SCALE_OUTER_VEC},
        {"i32 C32I outer vec", CUDA_R_32I, CUBLAS_COMPUTE_32I, CUDA_R_32F,
         SCALE_OUTER_VEC},
        {"bf16 C32F scalar", CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         SCALE_SCALAR},
        {"bf16 C32I scalar", CUDA_R_16BF, CUBLAS_COMPUTE_32I, CUDA_R_32F,
         SCALE_SCALAR},
        {"bf16 C32F vec128", CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F,
         SCALE_VEC128},
    };
    const size_t variant_count = sizeof(variants) / sizeof(*variants);

    /* Correctness problem: small, odd sizes, full int8 range. */
    const int m = 64, n = 30, k = 256; /* n = 2 mod 4, like 1870 tokens */
    int8_t *a_host = (int8_t *)malloc((size_t)k * m);
    int8_t *b_host = (int8_t *)malloc((size_t)k * n);
    float *m_scales = (float *)malloc((size_t)m * sizeof(float));
    float *n_scales = (float *)malloc((size_t)n * sizeof(float));
    srand(42);
    for (size_t i = 0; i < (size_t)k * m; i++)
        a_host[i] = (int8_t)(rand() % 255 - 127);
    for (size_t i = 0; i < (size_t)k * n; i++)
        b_host[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < m; i++) m_scales[i] = 1e-4f * (1 + rand() % 100);
    for (int i = 0; i < n; i++) n_scales[i] = 1e-3f * (1 + rand() % 100);
    double *exact = (double *)malloc((size_t)m * n * sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            int64_t sum = 0;
            for (int t = 0; t < k; t++)
                sum += (int64_t)a_host[(size_t)i * k + t] *
                       b_host[(size_t)j * k + t];
            exact[(size_t)j * m + i] = (double)sum;
        }
    void *a = upload(a_host, (size_t)k * m);
    void *b = upload(b_host, (size_t)k * n);

    printf("\n%-24s %-16s %-6s %s\n", "variant", "heuristic", "algos",
           "result vs host (worst relative error; which scaling it matches)");
    int supported[64] = {0};
    for (size_t vi = 0; vi < variant_count; vi++) {
        const variant *v = &variants[vi];
        plan p;
        cublasStatus_t status =
            plan_create(&p, v, m, n, k, m_scales, n_scales, workspace_bytes);
        cublasLtMatmulHeuristicResult_t results[16];
        int found = 0;
        if (status == CUBLAS_STATUS_SUCCESS)
            status = cublasLtMatmulAlgoGetHeuristic(lt, p.operation, p.a, p.b,
                                                    p.d, p.d, p.preference, 16,
                                                    results, &found);
        if (status != CUBLAS_STATUS_SUCCESS || !found) {
            printf("%-24s %-16s %-6d\n", v->name, status_name(status), found);
            plan_destroy(&p);
            continue;
        }
        size_t d_bytes = (size_t)m * n * element_bytes(v->output);
        void *d = NULL;
        CHECK(cudaMalloc(&d, d_bytes));
        status = cublasLtMatmul(lt, p.operation, alpha_of(&p, v), a, p.a, b,
                                p.b, beta_of(&p, v), d, p.d, d, p.d,
                                &results[0].algo, workspace, workspace_bytes,
                                0);
        CHECK(cudaDeviceSynchronize());
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("%-24s %-16s %-6d matmul %s\n", v->name, "ok", found,
                   status_name(status));
            cudaFree(d);
            plan_destroy(&p);
            continue;
        }
        void *d_host = malloc(d_bytes);
        CHECK(cudaMemcpy(d_host, d, d_bytes, cudaMemcpyDeviceToHost));
        /* Score the output against each interpretation; report the best. */
        const char *labels[] = {"unscaled", "row*col scales", "col scale",
                                "scalar scales"};
        double worst[4] = {0, 0, 0, 0};
        int saturated = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                size_t index = (size_t)j * m + i;
                double got = read_output(d_host, index, v->output);
                double e = exact[index];
                double want[4] = {e, e * m_scales[i] * n_scales[j],
                                  e * m_scales[i],
                                  e * m_scales[0] * n_scales[0]};
                if (v->output == CUDA_R_8I && fabs(e) > 127) saturated = 1;
                for (int w = 0; w < 4; w++) {
                    double denominator = fabs(want[w]) > 1e-3 ? fabs(want[w])
                                                              : 1e-3;
                    double error = fabs(got - want[w]) / denominator;
                    if (error > worst[w]) worst[w] = error;
                }
            }
        int best = 0;
        for (int w = 1; w < 4; w++)
            if (worst[w] < worst[best]) best = w;
        printf("%-24s %-16s %-6d %.2e %s%s\n", v->name, "ok", found,
               worst[best], labels[best],
               saturated ? " (i8 output saturates: not a usable result)" : "");
        supported[vi] = worst[best] < 1e-2 && !saturated;
        free(d_host);
        cudaFree(d);
        plan_destroy(&p);
    }
    cudaFree(a);
    cudaFree(b);

    /* Timing at the DiT's widest shapes, fox-fast token count, for every
     * variant that produced a correct result. Every returned algorithm is
     * timed, because the first heuristic pick is not always the fastest. */
    typedef struct {
        const char *name;
        int m, n, k;
    } shape;
    const shape shapes[] = {
        {"qkv  m=21504 n=1894 k=5376", 21504, 1894, 5376},
        {"fc1  m=28672 n=1894 k=5376", 28672, 1894, 5376},
        {"fc2  m=5376  n=1894 k=14336", 5376, 1894, 14336},
    };
    for (size_t si = 0; si < sizeof(shapes) / sizeof(*shapes); si++) {
        const shape *s = &shapes[si];
        printf("\n%s\n", s->name);
        void *big_a = NULL, *big_b = NULL;
        CHECK(cudaMalloc(&big_a, (size_t)s->k * s->m));
        CHECK(cudaMalloc(&big_b, (size_t)s->k * s->n));
        /* Random operands: constant bytes run INT8 GEMM ~1.4x faster than
         * real data on SM120, which would also skew the algorithm ranking. */
        {
            size_t bytes = (size_t)s->k * (s->m > s->n ? s->m : s->n);
            int8_t *fill = (int8_t *)malloc(bytes);
            for (size_t i = 0; i < bytes; i++)
                fill[i] = (int8_t)(rand() % 255 - 127);
            CHECK(cudaMemcpy(big_a, fill, (size_t)s->k * s->m,
                             cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(big_b, fill + 7, (size_t)s->k * s->n - 7,
                             cudaMemcpyHostToDevice));
            free(fill);
        }
        float *ms = (float *)malloc((size_t)s->m * sizeof(float));
        float *ns = (float *)malloc((size_t)s->n * sizeof(float));
        for (int i = 0; i < s->m; i++) ms[i] = 1e-4f;
        for (int i = 0; i < s->n; i++) ns[i] = 1e-3f;
        for (size_t vi = 0; vi < variant_count; vi++) {
            if (!supported[vi]) continue;
            const variant *v = &variants[vi];
            plan p;
            if (plan_create(&p, v, s->m, s->n, s->k, ms, ns,
                            workspace_bytes) != CUBLAS_STATUS_SUCCESS) {
                plan_destroy(&p);
                continue;
            }
            cublasLtMatmulHeuristicResult_t results[16];
            int found = 0;
            cublasLtMatmulAlgoGetHeuristic(lt, p.operation, p.a, p.b, p.d, p.d,
                                           p.preference, 16, results, &found);
            void *d = NULL;
            CHECK(cudaMalloc(&d, (size_t)s->m * s->n *
                                     element_bytes(v->output)));
            printf("  %-24s", v->name);
            for (int r = 0; r < found; r++) {
                int ok = 1;
                for (int warm = 0; warm < 3 && ok; warm++)
                    ok = cublasLtMatmul(lt, p.operation, alpha_of(&p, v),
                                        big_a, p.a, big_b, p.b, beta_of(&p, v),
                                        d, p.d, d, p.d, &results[r].algo,
                                        workspace, workspace_bytes, 0) ==
                         CUBLAS_STATUS_SUCCESS;
                if (!ok) {
                    printf(" [%d fail]", r);
                    continue;
                }
                CHECK(cudaDeviceSynchronize());
                const int iterations = 30;
                cudaEvent_t start, stop;
                CHECK(cudaEventCreate(&start));
                CHECK(cudaEventCreate(&stop));
                CHECK(cudaEventRecord(start));
                for (int it = 0; it < iterations; it++)
                    cublasLtMatmul(lt, p.operation, alpha_of(&p, v), big_a,
                                   p.a, big_b, p.b, beta_of(&p, v), d, p.d, d,
                                   p.d, &results[r].algo, workspace,
                                   workspace_bytes, 0);
                CHECK(cudaEventRecord(stop));
                CHECK(cudaEventSynchronize(stop));
                float ms_elapsed = 0;
                CHECK(cudaEventElapsedTime(&ms_elapsed, start, stop));
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                printf(" [%d %.3f ms]", r, ms_elapsed / iterations);
            }
            printf("\n");
            cudaFree(d);
            plan_destroy(&p);
        }
        free(ms);
        free(ns);
        cudaFree(big_a);
        cudaFree(big_b);
    }
    cudaFree(workspace);
    cublasLtDestroy(lt);
    return 0;
}
