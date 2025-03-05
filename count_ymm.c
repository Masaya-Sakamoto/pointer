#include <stdio.h>
#include <signal.h>
#include <setjmp.h>

// グローバルなジャンプバッファ
static jmp_buf jmpbuf;

// SIGILL（不正命令）発生時のシグナルハンドラ
void sigill_handler(int signum) {
    longjmp(jmpbuf, 1);
}

// マクロを用いて各YMMレジスタ用のテスト関数を定義
// ※ %% を使ってアセンブリ内の % をエスケープしています
#define TEST_YMM(n) \
__attribute__((noinline)) void test_ymm##n(void) { \
    __asm__ volatile("vmovdqa32 %%ymm" #n ", %%ymm" #n :::); \
}

// YMM0～YMM31の関数を生成
TEST_YMM(0)
TEST_YMM(1)
TEST_YMM(2)
TEST_YMM(3)
TEST_YMM(4)
TEST_YMM(5)
TEST_YMM(6)
TEST_YMM(7)
TEST_YMM(8)
TEST_YMM(9)
TEST_YMM(10)
TEST_YMM(11)
TEST_YMM(12)
TEST_YMM(13)
TEST_YMM(14)
TEST_YMM(15)
TEST_YMM(16)
TEST_YMM(17)
TEST_YMM(18)
TEST_YMM(19)
TEST_YMM(20)
TEST_YMM(21)
TEST_YMM(22)
TEST_YMM(23)
TEST_YMM(24)
TEST_YMM(25)
TEST_YMM(26)
TEST_YMM(27)
TEST_YMM(28)
TEST_YMM(29)
TEST_YMM(30)
TEST_YMM(31)

// 型定義：テスト関数へのポインタ
typedef void (*test_func_t)(void);

int main(void) {
    int count = 0;
    // SIGILL発生時のハンドラを登録
    signal(SIGILL, sigill_handler);

    // 各YMMレジスタのテスト関数を配列に登録
    test_func_t tests[32] = {
        test_ymm0,  test_ymm1,  test_ymm2,  test_ymm3,
        test_ymm4,  test_ymm5,  test_ymm6,  test_ymm7,
        test_ymm8,  test_ymm9,  test_ymm10, test_ymm11,
        test_ymm12, test_ymm13, test_ymm14, test_ymm15,
        test_ymm16, test_ymm17, test_ymm18, test_ymm19,
        test_ymm20, test_ymm21, test_ymm22, test_ymm23,
        test_ymm24, test_ymm25, test_ymm26, test_ymm27,
        test_ymm28, test_ymm29, test_ymm30, test_ymm31
    };

    // 各テスト関数を呼び出して、例外が起きなければカウントする
    for (int i = 0; i < 32; i++) {
        if (setjmp(jmpbuf) == 0) {
            tests[i]();  // 該当のYMMレジスタを参照する命令を実行
            count++;     // 正常に実行できた場合はカウントアップ
        } else {
            // SIGILLが発生した場合は、longjmpでここに戻りカウントしない
        }
    }

    printf("使用可能なYMMレジスタの本数: %d\n", count);
    return 0;
}
