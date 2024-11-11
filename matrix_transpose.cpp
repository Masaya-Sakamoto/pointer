#include <algorithm>
#include <immintrin.h> // AVX命令を使うために必要
#include <stdio.h>
#include <stdlib.h>

static const int WIDTH = 30720;
static const int HEIGHT = 152;

void memcpy_transpose_avx_blocked(float *A, float *B, int M, int N,
                                  int block_size) {
  // AはMxN行列、Bは転置後の行列（NxM）
  for (int i = 0; i < M; i += block_size) {
    for (int j = 0; j < N; j += block_size) {
      // 内部のブロックを転置
      for (int ii = i; ii < i + block_size && ii < M; ++ii) {
        for (int jj = j; jj < j + block_size && jj < N;
             jj += 4) { // AVXの128-bitレジスタで8個のfloatを処理
          // 8個ずつのデータを転送する
          __m128 data = _mm_loadu_ps(&A[ii * N + jj]);
          _mm_storeu_ps(&B[jj * M + ii], data);
        }
      }
    }
  }
}

void transpose_avx(const float *A, float *A_T, int WIDTH, int HEIGHT) {
  int block_size = 8;

  for (int b = 0; b < WIDTH; b += block_size) {
    for (int y = 0; y < HEIGHT; y++) {
      for (int i = 0; i < block_size; i++) {
        int x = std::min(b + i, WIDTH - 1);
        // AVXを用いて8つの要素を一度に処理
        if (x + 7 < WIDTH) {
          __m256 row = _mm256_loadu_ps(&A[y * WIDTH + x]); // y行のx列からロード
          _mm256_storeu_ps(&A_T[x * HEIGHT + y], row); // 転置して保存
        } else {
          // 残りの要素を1つずつ処理（端処理）
          for (int j = 0; j < WIDTH - x; j++) {
            A_T[(x + j) * HEIGHT + y] = A[y * WIDTH + x + j];
          }
        }
      }
    }
  }
}

int min(int a, int b) { return a >= b ? b : a; }

// void memcpy_transpose_blocked(float *A, float *B, int M, int N,
//                               int block_size) {
//   for (int b = 0; b < N; b += block_size) {
//     for (int y = 0; y < M; y++) {
//       for (int i = 0; i < block_size; i++) {
//         int x = min(b + i, M - 1);
//         B[x][y] = A[y][x];
//         // printf("[%d, %d]\n", x, y);
//       }
//     }
//   }
// }

int main() {
  int block_size = 4;
  float A[HEIGHT][WIDTH];
  float B[HEIGHT][WIDTH];

  // INIT A
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
      A[i][j] = i * WIDTH + j;
    }
  }

  // ブロックサイズ 4で転置を実行
  // memcpy_transpose_avx_blocked((float *)A, (float *)B, M, N, block_size);
  for (int b = 0; b < HEIGHT; b += block_size) {
    for (int y = 0; y < WIDTH; y++) {
      for (int i = 0; i < block_size; i++) {
        int x = min(b + i, HEIGHT - 1);
        B[x][y] = A[y][x];
        // printf("[%d, %d]\n", x, y);
      }
    }
  }

  // transpose_avx((float *)A, (float *)B, WIDTH, HEIGHT);
  // 転置後の行列を表示
  printf("Transposed Matrix B (8x8):\n");
  for (int i = 0; i < HEIGHT; i++) {
    for (int j = 0; j < WIDTH; j++) {
      printf("%3d ", (int)B[i][j]);
    }
    printf("\n");
  }

  return 0;
}
