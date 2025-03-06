#include <cblas.h>
#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

// #define DEBUG

#define M 8000
#define N 30720
#define K 30000
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
    float *A, *B, *C, *h_C;
    float *warmupA, *warmupB, *warmupC, *h_warmupC;
    A = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    B = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));
    h_C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float)); // for dump
    warmupA = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    warmupB = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    warmupC = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));
    h_warmupC = (float *)aligned_alloc(ALIGN, M * N * sizeof(float)); // for dump
    // Initialize array
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * K, warmupA);
    initArray(K * N, warmupB);
    // initArray(M * N, C);
    // initArray(M * N, h_C);
    memset(C, 0, M * N * sizeof(float));
    memset(h_C, 0, M * N * sizeof(float));
    memset(warmupC, 0, M * N * sizeof(float));
    memset(h_warmupC, 0, M * N * sizeof(float));

    // gpu initialization
    // Initialize cuBLAS context.
    cublasHandle_t handle;
    cublasCreate(&handle);
    // memcpy to device
    float *d_A, *d_B, *d_C;
    const float d_alpha = 1.0f;
    const float d_beta = 0.0f;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // gpu warm up run
    // cublasSetMatrix(); // 　cudamemcpyの代わりにこれを使うと良いかも
    cudaMemcpy(d_A, warmupA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_warmupC, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, warmupB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    cudaMemcpy(h_warmupC, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    std::cout << "CPU: calculation";
    fflush(stdout);

    auto time_point = std::chrono::high_resolution_clock::now();
#endif
    // cblas
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
#ifdef DEBUG
    auto duration = std::chrono::high_resolution_clock::now() - time_point;
    auto count = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    std::cout << " done!  " << count << " micro seconds" << std::endl;
#endif

#ifdef DEBUG
    std::cout << "GPU: calculation";
    fflush(stdout);
    auto time_point_gpu = std::chrono::high_resolution_clock::now();
#endif
    cudaMemcpyAsync(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    // memcpy to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
#ifdef DEBUG
    auto duration_gpu = std::chrono::high_resolution_clock::now() - time_point_gpu;
    auto count_gpu = std::chrono::duration_cast<std::chrono::microseconds>(duration_gpu).count();
    std::cout << " done!  " << count_gpu << " micro seconds" << std::endl;
#endif

    // comparison C and h_C
    // for (int m = 0; m < M; m++)
    // {
    //     for (int n = 0; n < N; n++)
    //     {
    //         int i = m * N + n; // Calculate the index in h_C
    //         float diff = C[i] - h_C[i] > 0 ? C[i] - h_C[i] : h_C[i] - C[i];
    //         if (diff > 8e-5)
    //         {
    //             std::cout << "i: " << i << ",  C[i]: " << C[i] << ",  h_C[i]: " << h_C[i] << ",  diff: " << diff
    //                       << std::endl;
    //         }
    //     }
    // }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // free host memory
    free(A);
    free(B);
    free(C);
    free(h_C);

    return 0;
}