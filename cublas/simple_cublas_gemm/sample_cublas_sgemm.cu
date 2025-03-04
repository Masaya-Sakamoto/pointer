#include <cblas.h>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

#define M 3
#define N 4
#define K 2
#define ALIGN 64

template <typename T> void initArray(size_t elements, T *array)
{
    srand(0); // Seed for reproducibility
    for (size_t i = 0; i < elements; ++i)
    {
        array[i] = static_cast<T>(rand()) / RAND_MAX;
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
    initArray(M * N, C);
    initArray(M * N, h_C);

    // cblas
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);

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
    cublasSgemm(
        handle, 
        CUBLAS_OP_T, CUBLAS_OP_T, // A, B 両方を転置
        N,                                // 結果行列の行数（実際はB^Tの行数）
        M,                                // 結果行列の列数（実際はA^Tの列数）
        K,                                // 内部次元
        &d_alpha,
        d_B, N,                 // d_B: 元のB。転置後は(N x K)となるのでリーディングディメンジョンはN
        d_A, K,                           // d_A: 元のA。転置後は(K x M)となるのでリーディングディメンジョンはK
        &d_beta,
        d_C, N); // d_C: 結果を受け取る行列。row-majorでMxNの行列の場合、転置結果はNxMとなりリーディングディメンジョンはN

    // memcpy to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // comparison C and h_C
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            int i = m * N + n; // Calculate the index in h_C
            std::cout << "i: " << i << ",  C[i]: " << C[i] << ",  h_C[i]: " << h_C[i] << ",  diff: " << C[i] - h_C[i] << std::endl;
        }
    }
    /*
    計算結果の出力
    i: 0,  C[i]: 1.02081,  h_C[i]: 1.02081,  diff: 0
    i: 1,  C[i]: 0.690894,  h_C[i]: 0.690894,  diff: 0
    i: 2,  C[i]: 0.735861,  h_C[i]: 0.735861,  diff: 0
    i: 3,  C[i]: 1.29546,  h_C[i]: 1.29546,  diff: 0
    i: 4,  C[i]: 1.03674,  h_C[i]: 1.03674,  diff: 0
    i: 5,  C[i]: 0.770977,  h_C[i]: 0.770977,  diff: 0
    i: 6,  C[i]: 0.923688,  h_C[i]: 0.923688,  diff: 0
    i: 7,  C[i]: 0.539635,  h_C[i]: 0.539635,  diff: 0
    i: 8,  C[i]: 0.752937,  h_C[i]: 0.752937,  diff: 0
    i: 9,  C[i]: 0.895035,  h_C[i]: 0.895035,  diff: 0
    i: 10,  C[i]: 0.832561,  h_C[i]: 0.832561,  diff: 0
    i: 11,  C[i]: 0.414277,  h_C[i]: 0.414277,  diff: 0
    */

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