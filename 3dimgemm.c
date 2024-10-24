#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

#define H0 2
#define H1 2
#define N 5
#define L 3

int main() {
    float ***A3;
    float ***B3;
    float *A;
    float *B;

    A3 = (float ***)malloc(sizeof(float)*L);
    for (int l = 0; l < L; l++) {
        A3[l] = (float **)malloc(sizeof(float)*H0);
        for (int h0 = 0; h0 < H0; h0++) {
            A3[l][h0] = (float *)malloc(sizeof(float)*H1);
            for (int h1 = 0; h1 < H1; h1++) {
                A3[l][h0][h1] = L*l + H0*h0 + h1;
            }
        }
    }

    B3 = (float ***)malloc(sizeof(float)*N);
    for (int n = 0; n < N; n++) {
        B3[n] = (float **)malloc(sizeof(float)*H0);
        for (int h0 = 0; h0 < H0; h0++) {
            B3[n][h0] = (float *)malloc(sizeof(float)*L);
            for (int l = 0; l < L; l++) {
                B3[n][h0][l] = N*n + H0*h0 + l;
            }
        }
    }

    A = (float *)A3;
    B = (float *)B3;

    cblas_sgemm
}