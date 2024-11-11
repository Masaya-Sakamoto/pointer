#include <cuda_runtime.h>
#include <iostream>

__global__ void addKernel(int* c, const int* a, const int* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// ホストから呼び出される関数
extern "C" void addVectors(int* c, const int* a, const int* b, int size) {
    int* d_a;
    int* d_b;
    int* d_c;

    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<<<(size + 255) / 256, 256>>>(d_c, d_a, d_b, size);

    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/*
cuda shared library compile
nvcc -shared -o libcuda_add.so kernel.cu
*/
