#include <cublas_v2.h>
#include <cblas.h>
#include <cstdlib>
#include <iostream>

#define M 3
#define N 4
#define K 2
#define ALIGN 64

template <typename T> 
void initArray(size_t elements, T *array)
{
    srand(0); // Seed for reproducibility
    for (size_t i = 0; i < elements; ++i) {
        array[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

int main() {
    float *A, *B, *C, *h_C;
    A = (float *)aligned_alloc(ALIGN, M*K*sizeof(float));
    B = (float *)aligned_alloc(ALIGN, K*N*sizeof(float));
    C = (float *)aligned_alloc(ALIGN, M*N*sizeof(float));
    h_C = (float *)aligned_alloc(ALIGN, M*N*sizeof(float)); // for dump
    // Initialize array
    initArray(M*K, A);
    initArray(K*N, B);
    initArray(M*N, C);
    initArray(M*N, h_C);

    // cblas
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);

    // Initialize cuBLAS context.
    cublasHandle_t handle;
    cublasCreate(&handle);

    // memcpy to device
    float *d_A, *d_B, *d_C;
    const float d_alpha = 1.0f;
    const float d_beta = 0.0f;
    cudaMalloc((void **)&d_A, M*K*sizeof(float));
    cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, K*N*sizeof(float));
    cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, M*N*sizeof(float));
    cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);
    // cuBLAS sgemm
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K, &d_alpha, d_A, M, d_B, K, &d_beta, d_C, M);
    // memcpy to host
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // comparison C and h_C
    for (int i = 0; i < M*N; ++i) {
        if (fabs(C[i] - h_C[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i  << ": expected " << C[i] << ", got " << h_C[i] << std::endl;
        }
    }

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