#include <cblas.h>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

#define M 3
#define N 4
#define K 2
#define ALIGN 64

#define IDX(i, j, ld) ((i) * (ld) + (j)) // Row-major index macro

template <typename T> void initArray(size_t elements, T *array)
{
    srand(0); // Seed for reproducibility
    for (size_t i = 0; i < elements; ++i)
    {
        array[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

void print_matrix(const float *A, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << A[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    float *A, *B, *C, *h_C;
    A = (float *)aligned_alloc(ALIGN, M * K * sizeof(float));
    B = (float *)aligned_alloc(ALIGN, K * N * sizeof(float));
    C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float));
    h_C = (float *)aligned_alloc(ALIGN, M * N * sizeof(float)); // for dump
    // Initialize array
    initArray(M * K, A);
    initArray(K * N, B);
    // initArray(M * N, C);
    // initArray(M * N, h_C);
    memset(C, 0, M * N * sizeof(float));
    memset(h_C, 0, M * N * sizeof(float));

    // cblas
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    // Initialize cuBLAS context.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // memcpy to device
    float *d_A, *d_B, *d_C;
    const float d_alpha = 1.0f;
    const float d_beta = 0.0f;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, M * N * sizeof(float));
    cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    // cuBLAS sgemm
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &d_alpha, d_B, N, d_A, K, &d_beta, d_C, N);

    // memcpy to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // comparison C and h_C
    std::cout << "CBLAS result (row-major):\n";
    print_matrix(C, M, N);

    std::cout << "cuBLAS result (row-major interpreted):\n";
    print_matrix(h_C, M, N);

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