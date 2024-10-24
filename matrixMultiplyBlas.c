#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 行列サイズ
    int m = 4;   // 行数
    int n = 4;   // 列数
    int kl = 1;  // 下バンド幅
    int ku = 1;  // 上バンド幅

    // バンド行列 A のデータ
    // float A[16] = {
    //     1,  2,  0,  0,
    //     3,  4,  5,  0,
    //     6,  7,  8,  9,
    //     10, 11, 12, 13
    // };
    float A[16] = {
        1, 2, 0, 0, 
        3, 4, 5, 0, 
        0, 7, 8, 9, 
        0, 0, 12, 13};

    // ベクトル X のデータ
    float X[4] = {1, 2, 3, 4};

    // ベクトル Y のデータ（初期化）
    float Y[4] = {0, 0, 0, 0};

    // スカラー値
    float alpha = 1.0;
    float beta = 0.0;

    // BLASのsgbmvを呼び出し
    cblas_sgbmv(CblasRowMajor, CblasNoTrans, m, n, kl, ku, alpha, A, 2 * kl + n,
                X, 1, beta, Y, 1);

    // 結果の出力
    printf("Resulting vector Y:\n");
    for (int i = 0; i < m; ++i) {
    printf("%f\n", Y[i]);
    }

    return 0;
}