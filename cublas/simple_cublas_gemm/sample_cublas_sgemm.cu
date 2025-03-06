#include <chrono>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

#define DEBUG
#define CBLAS

#ifdef CBLAS
#include <cblas.h>
#endif

#define M 2
#define N 30720
#define K 304
#define ALIGN 64

void initArray(size_t elements, float *array)
{
    srand(0);
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
    float *h_A, *h_B, *h_C;

    // Allocate aligned memory for better performance
    A = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    B = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));

    // Allocate pinned host memory for efficient cudaMemcpy
    cudaHostAlloc((void **)&h_A, M * K * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, K * N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, M * N * sizeof(float), cudaHostAllocDefault);

    // Initialize arrays
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * N, C);

    // Copy data to host pinned memory
    memcpy(h_A, A, M * K * sizeof(float));
    memcpy(h_B, B, K * N * sizeof(float));
    memcpy(h_C, C, M * N * sizeof(float));

    // Initialize cuBLAS context and stream
    cublasHandle_t handle;
    cudaStream_t stream;

    cudaStreamCreate(&stream);
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    const float d_alpha = 1.0f;
    const float d_beta = 0.0f;

    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // GPU warm-up run
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice, stream);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    cudaStreamSynchronize(stream);

    // CPU and GPU computation timing
#ifdef DEBUG
    std::cout << "CPU: calculation ";
    fflush(stdout);

    auto time_point = std::chrono::high_resolution_clock::now();
#endif
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

#ifdef DEBUG
    auto duration = std::chrono::high_resolution_clock::now() - time_point;
    auto count = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    std::cout << "done!   " << count << " micro seconds" << std::endl;
#endif

    // Re-initialize arrays for GPU computation
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * N, C);

    memcpy(h_A, A, M * K * sizeof(float));
    memcpy(h_B, B, K * N * sizeof(float));
    memcpy(h_C, C, M * N * sizeof(float));

    // Copy data to device
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice, stream);

#ifdef DEBUG
    std::cout << "GPU: calculation ";
    fflush(stdout);
    auto time_point_gpu = std::chrono::high_resolution_clock::now();
#endif

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Ensure all operations are completed

#ifdef DEBUG
    auto duration_gpu = std::chrono::high_resolution_clock::now() - time_point_gpu;
    auto count_gpu = std::chrono::duration_cast<std::chrono::microseconds>(duration_gpu).count();
    std::cout << "done!   " << count_gpu << " micro seconds" << std::endl;
#endif

    // // Compare element-wise with tolerance
    // for (int i = 0; i < M; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         float cblas_val = C[i * N + j];
    //         float cublas_val = h_C[i * N + j];

    //         // Use a tolerance to compare floating-point values
    //         if (fabsf(cblas_val - cublas_val) > 1e-6f)
    //         {
    //             printf("Mismatch at position (%d, %d): CBLAS=%.4f vs cuBLAS=%.4f\n", i + 1, j + 1, cblas_val,
    //                    cublas_val);
    //         }
    //     }
    // }

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

    return 0;
}