#include <omp.h>
#include <stdio.h>

int main() {
    int num_threads = omp_get_max_threads();  // 最大スレッド数を取得
    printf("Default OMP_NUM_THREADS: %d\n", num_threads);
    return 0;
}
