#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

#define DEBUG
#define INFO
#define CBLAS

#ifdef CBLAS
#include <cblas.h>
#endif

#define M 2
#define N 61440
#define K 3200
#define ALIGN 64

void initArray(size_t elements, float *array)
{
    srand(0); // Seed for reproducibility
    for (size_t i = 0; i < elements; ++i)
    {
        array[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void print_matrix(const float *A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << A[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    float *A, *B, *C;
    float *t_A, *t_B, *t_C; // for cblas(cblas_col_major)
    float *h_A, *h_B, *h_C;

    // Allocate main memory (aligned)
    A = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    B = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));
    t_A = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    t_B = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    t_C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));

    // Allocate page-locked host memory
    cudaHostAlloc((void **)&h_A, M * K * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, K * N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, M * N * sizeof(float), cudaHostAllocDefault);

    // Initialize arrays
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * N, C);

    // Copy to page-locked host buffers
    memcpy(t_A, A, M * K * sizeof(float));
    memcpy(t_B, B, K * N * sizeof(float));
    memcpy(t_C, C, M * N * sizeof(float));
    memcpy(h_A, A, M * K * sizeof(float));
    memcpy(h_B, B, K * N * sizeof(float));
    memcpy(h_C, C, M * N * sizeof(float));

    // GPU initialization
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream); // Associate cuBLAS with stream

    float *d_A, *d_B, *d_C;
    const float d_alpha = 1.0f;
    const float d_beta = 0.0f;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // Warm-up: Async data transfer and SGEMM
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    cudaStreamSynchronize(stream);

    // Reinitialize data
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * N, C);
    memcpy(t_A, A, M * K * sizeof(float));
    memcpy(t_B, B, K * N * sizeof(float));
    memcpy(t_C, C, M * N * sizeof(float));
    memcpy(h_A, A, M * K * sizeof(float));
    memcpy(h_B, B, K * N * sizeof(float));
    memcpy(h_C, C, M * N * sizeof(float));

#ifdef DEBUG
    // CPU timing
    std::cout << "CPU: calculation";
    fflush(stdout);
    auto cpu_start = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#endif
    auto cpu_duration = std::chrono::high_resolution_clock::now() - cpu_start;
    auto cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_duration).count();
    std::cout << " done!  " << cpu_us << " μs\n";
#endif

#ifdef DEBUG
    // CPU timing
    std::cout << "CPU: calculation";
    fflush(stdout);
    cpu_start = std::chrono::high_resolution_clock::now();
#ifdef DEBUG
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K, 1.0f, t_B, N, t_A, K, 0.0f, t_C, N);
#endif
    cpu_duration = std::chrono::high_resolution_clock::now() - cpu_start;
    cpu_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_duration).count();
    std::cout << " done!  " << cpu_us << " μs\n";
#endif

#ifdef DEBUG
    // GPU timing (kernel + result copy-back)
    std::cout << "GPU: calculation";
    fflush(stdout);
    auto gpu_start = std::chrono::high_resolution_clock::now();
#endif

    // Pre-copy all data to GPU (excluded from timing)
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

#ifdef DEBUG
    auto gpu_duration = std::chrono::high_resolution_clock::now() - gpu_start;
    auto gpu_us = std::chrono::duration_cast<std::chrono::microseconds>(gpu_duration).count();
    std::cout << " done!  " << gpu_us << " μs\n";
#endif

#ifdef DEBUG
    // Compare element-wise with tolerance
    bool results_match_gpu = true;
    bool results_match_cpuCol = true;
    bool results_match_gpu_cpuCol = true;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float cblas_val = C[i * N + j];
            float cblas_col_val = t_C[i * N + j];
            float cublas_val = h_C[i * N + j];

            // Use a tolerance to compare floating-point values
            if (fabsf(cblas_val - cublas_val) > 1e-3f)
            {
                results_match_gpu = false;
#ifdef INFO
                printf("Mismatch at position (%d, %d): CBLAS=%.4f vs cuBLAS=%.4f\n", i + 1, j + 1, cblas_val,
                       cublas_val);
#endif
            }
            if (fabsf(cblas_val - cblas_col_val) > 1e-3f)
            {
                results_match_cpuCol = false;
#ifdef INFO
                printf("Mismatch at position (%d, %d): CBLAS=%.4f vs CBLAS_COL=%.4f\n", i + 1, j + 1, cblas_val,
                       cublas_val);
#endif
            }
            if (fabsf(cblas_col_val - cublas_val) > 1e-3f)
            {
                results_match_gpu_cpuCol = false;
#ifdef INFO
                printf("Mismatch at position (%d, %d): CBLAS_COL=%.4f vs cuBLAS=%.4f\n", i + 1, j + 1, cblas_col_val,
                       cublas_val);
#endif
            }
        }
    }

    if (results_match_gpu)
    {
        printf("Results match!\n");
        for (int _chk = 0; _chk < 10; _chk++)
        {
            int chk_id = rand() % (N * M);
            printf("%3d, id=%9d, C=%7.4f, t_C=%7.4f\n", _chk, chk_id, C[chk_id], h_C[chk_id]);
        }
    }
    else
    {
        printf("Results do not match within tolerance.\n");
    }

    if (results_match_cpuCol)
    {
        printf("Results cpuCol match!\n");
        for (int _chk = 0; _chk < 10; _chk++)
        {
            int chk_id = rand() % (N * M);
            printf("%3d, id=%9d, C=%7.4f, t_C=%7.4f\n", _chk, chk_id, C[chk_id], t_C[chk_id]);
        }
    }
    else
    {
        printf("Results cpuCol do not match within tolerance.\n");
    }

    if (results_match_gpu_cpuCol)
    {
        printf("Results gpu_cpuCol match!\n");
        for (int _chk = 0; _chk < 10; _chk++)
        {
            int chk_id = rand() % (N * M);
            printf("%3d, id=%9d, C=%7.4f, t_C=%7.4f\n", _chk, chk_id, t_C[chk_id], h_C[chk_id]);
        }
    }
    else
    {
        printf("Results gpu_cpuCol do not match within tolerance.\n");
    }
#endif

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    free(A);
    free(B);
    free(C);
    free(t_A);
    free(t_B);
    free(t_C);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    return 0;
}