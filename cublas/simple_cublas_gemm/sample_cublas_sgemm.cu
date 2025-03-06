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
    float *h_A, *h_B, *h_C;

    // alloc main memory
    A = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    B = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));

    // alloc host memory for fast cudaMemcpy
    cudaHostAlloc((void **)&h_A, M * K * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_B, K * N * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_C, M * N * sizeof(float), cudaHostAllocDefault);

    // Initialize array
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * N, C);

    // memcpy X -> h_X
    memcpy(h_A, A, M * K * sizeof(float));
    memcpy(h_B, B, K * N * sizeof(float));
    memcpy(h_C, C, M * N * sizeof(float));

    // gpu initialization
    // Initialize cuBLAS context.
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    // After creating the stream, set it for cuBLAS
    cublasCreate(&handle);
    cublasSetStream(handle, stream); // Add this line

    // memcpy to device
    float *d_A, *d_B, *d_C;
    const float d_alpha = 1.0f;
    const float d_beta = 0.0f;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // gpu warm up run
    // cublasSetMatrix(); // 　cudamemcpyの代わりにこれを使うと良いかも
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);
    // Synchronize the stream to wait for SGEMM to complete
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
    // reinitialize A, B, C
    initArray(M * K, A);
    initArray(K * N, B);
    initArray(M * N, C);
    memcpy(h_A, A, M * K * sizeof(float));
    memcpy(h_B, B, K * N * sizeof(float));
    memcpy(h_C, C, M * N * sizeof(float));
    cudaMemcpyAsync(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice, stream);

// GPU timing section
#ifdef DEBUG
    std::cout << "GPU: calculation";
    fflush(stdout);
    auto time_point_gpu = std::chrono::high_resolution_clock::now();
#endif

    // Correct cuBLAS SGEMM call for row-major matrices
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);

    // Synchronize the stream to wait for SGEMM to complete
    cudaStreamSynchronize(stream);

    // Memcpy to host (this is blocking but timed after SGEMM)
    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);

#ifdef DEBUG
    cudaStreamSynchronize(stream); // Ensure all GPU operations are done
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
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    // free main memory
    free(A);
    free(B);
    free(C);

    return 0;
}