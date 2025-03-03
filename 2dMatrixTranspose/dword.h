#include "types.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

inline void _word2x2TransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ２個のリスト */,
                                    _word_t **outputPtrList /* 転置先小行列の先頭ポインタ２個のリスト */)
{
    // const dword_t upper_mask = (dword_t)0xFFFFFFFF00000000;
    // const dword_t lower_mask = (dword_t)0x00000000FFFFFFFF;
    // // dword2つ分をqword領域にコピーする
    // dword_t a1 = *(dword_t *)inputPtrList[0];
    // dword_t a2 = *(dword_t *)inputPtrList[1];
    // // 小行列の(0,1)要素の値を一時的に保存する変数
    // const _word_t tmp = (a1 & upper_mask) >> sizeof(_word_t) * BITS_PER_BYTE;
    // // a1 の上位32ビットとa2 の下位32ビットを交換する
    // a1 = ((a1 & lower_mask) | ((a2 & lower_mask) << _WORD_SIZE * BITS_PER_BYTE));
    // a2 = ((a2 & upper_mask) | (tmp));
    // // 転置した小行列を転送先にコピーする
    // *(dword_t *)outputPtrList[0] = a1;
    // *(dword_t *)outputPtrList[1] = a2;
    const _word_t tmp = inputPtrList[0][1];
    inputPtrList[0][1] = inputPtrList[1][0];
    inputPtrList[1][0] = tmp;
    *(dword_t *)outputPtrList[0] = *(dword_t *)inputPtrList[0];
    *(dword_t *)outputPtrList[1] = *(dword_t *)inputPtrList[1];
}

inline void dword_unpackl_epi32(dword_t *src1, dword_t *src2, dword_t *dst)
{
    // src1, src2のu64をu32で区切った際の偶数番目の値をdstに書き込む
    *dst = 0;
    *dst |= (*src1 && DWORD_MASK0_EACH32);
    *dst |= (*src2 && DWORD_MASK0_EACH32) << 32;
    return;
}

inline void dword_unpackl_epi16(dword_t *src1, dword_t *src2, dword_t *dst)
{
    // src1, src2のu64をu16で区切った際の偶数番目(0,0), (0,2), (1,0), (1,2)の値をdstに書き込む
    *dst = 0;
    *dst |= (*src1 & DWORD_MASK0_EACH16);
    *dst |= (*src2 & DWORD_MASK0_EACH16) << 16;
    *dst |= (*src1 & DWORD_MASK2_EACH16) << 32;
    *dst |= (*src2 & DWORD_MASK2_EACH16) << 48;
    return;
}

inline void dword_unpackl_epi8(dword_t *src1, dword_t *src2, dword_t *dst)
{
    // src1, src2のu64をu8で区切った際の偶数番目の値をdstに書き込む
    *dst = 0;
    *dst |= (*src1 & DWORD_MASK0_EACH8);
    *dst |= (*src2 & DWORD_MASK0_EACH8) << 8;
    *dst |= (*src1 & DWORD_MASK2_EACH8) << 16;
    *dst |= (*src2 & DWORD_MASK2_EACH8) << 24;
    *dst |= (*src1 & DWORD_MASK4_EACH8) << 32;
    *dst |= (*src2 & DWORD_MASK4_EACH8) << 40;
    *dst |= (*src1 & DWORD_MASK6_EACH8) << 48;
    *dst |= (*src2 & DWORD_MASK6_EACH8) << 56;
    return;
}

inline void dword_unpackh_epi32(dword_t *src1, dword_t *src2, dword_t *dst)
{
    // src1, src2のu64をu32で区切った際の奇数番目の値をdstに書き込む
    *dst = 0;
    *dst |= (*src1 && DWORD_MASK1_EACH32);
    *dst |= (*src2 && DWORD_MASK1_EACH32) << 32;
    return;
}

inline void dword_unpackh_epi16(dword_t *src1, dword_t *src2, dword_t *dst)
{
    // src1, src2のu64をu16で区切った際の奇数番目(0,0), (0,2), (1,0), (1,2)の値をdstに書き込む
    *dst = 0;
    *dst |= (*src1 & DWORD_MASK1_EACH16);
    *dst |= (*src2 & DWORD_MASK1_EACH16) << 16;
    *dst |= (*src1 & DWORD_MASK3_EACH16) << 32;
    *dst |= (*src2 & DWORD_MASK3_EACH16) << 48;
    return;
}

inline void dword_unpackh_epi8(dword_t *src1, dword_t *src2, dword_t *dst)
{
    // src1, src2のu64をu8で区切った際の奇数番目の値をdstに書き込む
    *dst = 0;
    *dst = 0;
    *dst |= (*src1 & DWORD_MASK1_EACH8);
    *dst |= (*src2 & DWORD_MASK1_EACH8) << 8;
    *dst |= (*src1 & DWORD_MASK3_EACH8) << 16;
    *dst |= (*src2 & DWORD_MASK3_EACH8) << 24;
    *dst |= (*src1 & DWORD_MASK5_EACH8) << 32;
    *dst |= (*src2 & DWORD_MASK5_EACH8) << 40;
    *dst |= (*src1 & DWORD_MASK7_EACH8) << 48;
    *dst |= (*src2 & DWORD_MASK7_EACH8) << 56;
    return;
}

inline void hword4x4TransposeKernel(hword_t **inputPtrList /* 元小行列の先頭ポインタ4個のリスト */,
                                    hword_t **outputPtrList /* 転置先小行列の先頭ポインタ4個のリスト */)
{
    // 4x4小行列の転置
    // (16bit x4) x4
    dword_t a0, a1, a2, a3;
    dword_t b0, b1, b2, b3;
    // load data and store lower and upper 16bit part
    a0 = *(dword_t *)inputPtrList[0];
    a1 = *(dword_t *)inputPtrList[1];
    dword_unpackl_epi16(&a0, &a1, &b0);
    dword_unpackh_epi16(&a0, &a1, &b2);
    a2 = *(dword_t *)inputPtrList[2];
    a3 = *(dword_t *)inputPtrList[3];
    dword_unpackl_epi16(&a2, &a3, &b1);
    dword_unpackh_epi16(&a2, &a3, &b3);

    // store lower and upper 32bit part
    // メモリ操作を分けたほうが早いかも。。
    dword_unpackl_epi32(&b0, &b1, (dword_t *)outputPtrList[0]);
    dword_unpackl_epi32(&b2, &b3, (dword_t *)outputPtrList[1]);
    dword_unpackh_epi32(&b0, &b1, (dword_t *)outputPtrList[2]);
    dword_unpackh_epi32(&b2, &b3, (dword_t *)outputPtrList[3]);
    return;
}

inline void char8x8TransposeKernel(char_t **inputPtrList /* 元小行列の先頭ポインタ8個のリスト */,
                                   char_t **outputPtrList /* 転置先小行列の先頭ポインタ8個のリスト */)
{
    dword_t a0, a1, a2, a3, a4, a5, a6, a7;
    dword_t b0, b1, b2, b3, b4, b5, b6, b7;

    // load data and sotre lower and upper 8bit part
    a0 = *(dword_t *)inputPtrList[0];
    a1 = *(dword_t *)inputPtrList[1];
    dword_unpackl_epi8(&a0, &a1, &b0);
    dword_unpackh_epi8(&a0, &a1, &b4);
    a2 = *(dword_t *)inputPtrList[2];
    a3 = *(dword_t *)inputPtrList[3];
    dword_unpackl_epi8(&a2, &a3, &b1);
    dword_unpackh_epi8(&a2, &a3, &b5);
    a4 = *(dword_t *)inputPtrList[4];
    a5 = *(dword_t *)inputPtrList[5];
    dword_unpackl_epi8(&a4, &a5, &b2);
    dword_unpackh_epi8(&a4, &a5, &b6);
    a6 = *(dword_t *)inputPtrList[6];
    a7 = *(dword_t *)inputPtrList[7];
    dword_unpackl_epi8(&a6, &a7, &b3);
    dword_unpackh_epi8(&a6, &a7, &b7);

    // store lower and upper 16bit part
    dword_unpackl_epi16(&b0, &b1, &a0);
    dword_unpackl_epi16(&b2, &b3, &a1);
    dword_unpackl_epi16(&b4, &b5, &a2);
    dword_unpackl_epi16(&b6, &b7, &a3);
    dword_unpackh_epi16(&b0, &b1, &a4);
    dword_unpackh_epi16(&b2, &b3, &a5);
    dword_unpackh_epi16(&b4, &b5, &a6);
    dword_unpackh_epi16(&b6, &b7, &a7);

    // store lower and upper 32bit part
    dword_unpackl_epi32(&a0, &a1, (dword_t *)outputPtrList[0]);
    dword_unpackl_epi32(&a2, &a3, (dword_t *)outputPtrList[1]);
    dword_unpackl_epi32(&a4, &a5, (dword_t *)outputPtrList[2]);
    dword_unpackl_epi32(&a6, &a7, (dword_t *)outputPtrList[3]);
    dword_unpackh_epi32(&a0, &a1, (dword_t *)outputPtrList[4]);
    dword_unpackh_epi32(&a2, &a3, (dword_t *)outputPtrList[5]);
    dword_unpackh_epi32(&a4, &a5, (dword_t *)outputPtrList[6]);
    dword_unpackh_epi32(&a6, &a7, (dword_t *)outputPtrList[7]);
}

void hword4x4Transpose(hword_t *src, hword_t *dst, size_t width, size_t height)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            dst[y * width + x] = src[x * height + y];
        }
    }
}
void char8x8Transpose(char_t *src, char_t *dst, size_t width, size_t height)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            dst[y * width + x] = src[x * height + y];
        }
    }
}