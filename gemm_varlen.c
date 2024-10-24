#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int m = 2;
    int n = 5;
    int N = 10;
    int k = 3;
    double alpha = 1.0;
    double beta = 0.0;
    // double A[m * k];
    double *A;
    A = (double *)malloc(sizeof(double) * m * k);
    for (int i = 0; i < m * k; i++)
        A[i] = i + 1;
    // double B[k * N];
    double *B;
    B = (double *)malloc(sizeof(double) * k * N);
    for (int i = 0; i < k * n; i++)
        B[i] = i + 1;
    // double C[m * N]; // x * y = 38 14 20 26
                        //         83 38 53 68 となることを確かめる
    double *C;
    C = (double *)malloc(sizeof(double) * m * N);

    int i, j;
    cblas_dgemm(CblasRowMajor, // 通常のCの行列であればCBlasRowMajorを指定
                CblasNoTrans,  // Aについて転置しない場合　CblasNoTrans 転置する場合　 CblasTrans
                CblasNoTrans,  // Bについて転置しない場合　CblasNoTrans 転置する場合　 CblasTrans
                m,             // Aの行数
                n,             // Bの列数
                k,             // Aの列数　＝　Bの行数（一致していないと掛け算できない）
                1,             // alpha の値
                A,             // A
                k,             // leading dimension (通常はAの列数)
                &B[3],             // B
                n,             // leading dimension (通常はBの列数)
                0,             // beta の値
                C,             // C
                n              // leading dimension (通常はCの列数)
    );

    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            // 二次元配列の場合でもこの書き方でOK
            printf("%lf ", C[i * n + j]); //(i,j)成分は i*ldC+jの形で書ける ldCはCのleading dimension :Cの列数
            // printf("%lf ",C[i][j]);//もちろん、二次元配列で定義すればこれでも動く
        }
        printf("\n");
    }

    return 0;
}