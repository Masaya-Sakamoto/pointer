#include <cblas.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "types.h"

#define LOOPS 1000
#define ALIGN_SIZE 64
#define RAND_MAX 2147483647

typedef struct inputParam
{
    cf_t *A;
    cf_t *B;
    cf_t *C;
    cf_t *noise_buffer;
    int buffer_size;
    int m;
    int n;
    int k;
} inputParam_t;

int conplexUrandGenerator(cf_t *noise_buffer, cf_t *ret_array, int buffer_size, int array_size, int seed)
{
    for (int i = 0; i < array_size; i++)
    {
        ret_array[i] = noise_buffer[(seed + buffer_size) % buffer_size];
        seed = (48271 * seed + 0x7FFFFFFF) % 0x7FFFFFFF;
    }
    return seed;
}

void calcBcolmajor(int m, int n, int k, cf_t alpha, cf_t beta, cf_t *A, cf_t *B, cf_t *C)
{
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, &alpha, A, k, B, k, &beta, C, n);
}

void calcBrowmajor(int m, int n, int k, cf_t alpha, cf_t beta, cf_t *A, cf_t *B, cf_t *C)
{
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, &alpha, A, k, B, n, &beta, C, n);
}

void *calcThreadColmajor(void *param)
{
    inputParam_t *inputParam = (inputParam_t *)param;
    static int seed = 0;
    const cf_t alpha = {1.0, 0.0};
    const cf_t beta = {0.0, 0.0};

    for (int i = 0; i < LOOPS; i++)
    {
        // init input matrix
        seed = conplexUrandGenerator(inputParam->noise_buffer, inputParam->A, inputParam->n * 20,
                                     inputParam->m * inputParam->k, seed);
        seed = conplexUrandGenerator(inputParam->noise_buffer, inputParam->B, inputParam->n * 20,
                                     inputParam->k * inputParam->n, seed);

        // calculation
        calcBcolmajor(inputParam->m, inputParam->n, inputParam->k, alpha, beta, inputParam->A, inputParam->B,
                      inputParam->C);
        if (i % (int)(LOOPS / 10) == 0)
        {
            printf("col major #%d finished\n", i);
            fflush(stdout);
        }
    }
}

void *calcThreadRowmajor(void *param)
{
    inputParam_t *inputParam = (inputParam_t *)param;
    static uint32_t seed = RAND_MAX / 2;
    const cf_t alpha = {1.0, 0.0};
    const cf_t beta = {0.0, 0.0};

    for (int i = 0; i < LOOPS; i++)
    {
        // init input matrix
        seed = conplexUrandGenerator(inputParam->noise_buffer, inputParam->A, inputParam->m * 20,
                                     inputParam->m * inputParam->k, seed);
        seed = conplexUrandGenerator(inputParam->noise_buffer, inputParam->B, inputParam->m * 20,
                                     inputParam->k * inputParam->n, seed);

        // calculation
        calcBrowmajor(inputParam->m, inputParam->n, inputParam->k, alpha, beta, inputParam->A, inputParam->B,
                      inputParam->C);
        if (i % (int)(LOOPS / 10) == 0)
        {
            printf("row major #%d finished\n", i);
            fflush(stdout);
        }
    }
}

int main()
{
    int nb_rx = 2;
    int nb_tx = 2;
    int nsamps = 30720 + 320;
    int channel_length = 152;

    int m = nb_rx;
    int n = nsamps;
    int k = nb_tx * channel_length;

    int buffer_size = n * 20;

    inputParam_t inputRow, inputCol;
    cf_t *Ar, *Ac, *Br, *Bc, *Cr, *Cc, *Nr, *Nc;

    pthread_t rowtherad, colthread;

    // noise generation
    float *noise_buff;
    noise_buff = (float *)aligned_alloc(ALIGN_SIZE, sizeof(float) * buffer_size);
    for (int i = 0; i < 2 * n; i++)
    {
        noise_buff[i] = ((float)rand() / RAND_MAX);
    }

    // input matrix allocation
    Ar = (cf_t *)aligned_alloc(ALIGN_SIZE, m * k * sizeof(cf_t));
    Ac = (cf_t *)aligned_alloc(ALIGN_SIZE, m * k * sizeof(cf_t));
    Br = (cf_t *)aligned_alloc(ALIGN_SIZE, k * n * sizeof(cf_t));
    Bc = (cf_t *)aligned_alloc(ALIGN_SIZE, k * n * sizeof(cf_t));
    Cr = (cf_t *)aligned_alloc(ALIGN_SIZE, m * n * sizeof(cf_t));
    Cc = (cf_t *)aligned_alloc(ALIGN_SIZE, m * n * sizeof(cf_t));
    Nr = (cf_t *)aligned_alloc(ALIGN_SIZE, buffer_size * sizeof(cf_t));
    Nc = (cf_t *)aligned_alloc(ALIGN_SIZE, buffer_size * sizeof(cf_t));

    // set values into noise_array
    for (int i = 0; i < n; i++)
    {
        Nr[i].r = noise_buff[i];
        Nr[i].i = noise_buff[n + i];
        Nc[i].r = noise_buff[i];
        Nc[i].i = noise_buff[n + i];
    }

    inputRow.A = Ar;
    inputRow.B = Br;
    inputRow.C = Cr;
    inputRow.m = m;
    inputRow.n = n;
    inputRow.k = k;
    inputRow.noise_buffer = Nr;
    inputRow.buffer_size = buffer_size;

    inputCol.A = Ac;
    inputCol.B = Bc;
    inputCol.C = Cc;
    inputCol.m = m;
    inputCol.n = n;
    inputCol.k = k;
    inputCol.noise_buffer = Nc;
    inputCol.buffer_size = buffer_size;

    // create thread
    // if (pthread_create(&rowtherad, NULL, calcThreadRowmajor, (void *)&inputRow) != 0)
    // {
    //     printf("failed to create calcThreadRowmajor thread\n");
    //     exit(1);
    // }
    if (pthread_create(&colthread, NULL, calcThreadColmajor, (void *)&inputCol) != 0)
    {
        printf("failed to create calcThreadColmajor thread\n");
        exit(1);
    }

    // wait for thread termination
    // if (pthread_join(rowtherad, NULL) != 0)
    // {
    //     printf("failed to join calcThreadRowmojor thread\n");
    //     exit(1);
    // }
    if (pthread_join(colthread, NULL) != 0)
    {
        printf("failed to join calcThreadColmajor thread\n");
        exit(1);
    }

    free(Ar);
    free(Ac);
    free(Br);
    free(Bc);
    free(Cr);
    free(Cc);
    free(Nr);
    free(Nc);
    free(noise_buff);

    return 0;
}