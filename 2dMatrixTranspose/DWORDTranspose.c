#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ↓↓↓↓↓↓↓↓↓↓ DEFINITIONS ↓↓↓↓↓↓↓↓↓↓

#define BITS_PER_BYTE 8
#define DWORD_SIZE 8
#define _WORD_SIZE 4
#define HWORD_SIZE 2
#define CHAR_SIZE 1
#define DWORD_MASK0_EACH32 0x00000000FFFFFFFF
#define DWORD_MASK1_EACH32 0xFFFFFFFF00000000
#define DWORD_MASK0_EACH16 0x000000000000FFFF
#define DWORD_MASK1_EACH16 0x00000000FFFF0000
#define DWORD_MASK2_EACH16 0x0000FFFF00000000
#define DWORD_MASK3_EACH16 0xFFFF000000000000
#define DWORD_MASK0_EACH8 0x00000000000000FF
#define DWORD_MASK1_EACH8 0x000000000000FF00
#define DWORD_MASK2_EACH8 0x0000000000FF0000
#define DWORD_MASK3_EACH8 0x00000000FF000000
#define DWORD_MASK4_EACH8 0x000000FF00000000
#define DWORD_MASK5_EACH8 0x0000FF0000000000
#define DWORD_MASK6_EACH8 0x00FF000000000000
#define DWORD_MASK7_EACH8 0xFF00000000000000

typedef uint64_t dword_t;
typedef uint32_t _word_t;
typedef uint16_t hword_t;
typedef uint8_t char_t;

/// @brief Transposes a 2D matrix of _word_ elements in a block-based manner for efficient processing.
///
/// This function transposes a 2D matrix by processing 2x2 blocks at a time to optimize performance.
/// If the matrix dimensions are not even, it handles the residual rows and columns separately.
///
/// @param src Pointer to the source _word_t array representing the input matrix.
/// @param dst Pointer to the destination _word_t array where the transposed matrix will be stored.
/// @param width The width (number of columns) of the matrix.
/// @param height The height (number of rows) of the matrix.
///
void DWORD2x2Transpose(_word_t *src, _word_t *dst, size_t width, size_t height);

// ↑↑↑↑↑↑↑↑↑↑↑ DEFINITIONS ↑↑↑↑↑↑↑↑↑↑

inline void _WORD2x2TransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ２個のリスト */,
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

void DWORD2x2Transpose(_word_t *src, _word_t *dst, size_t width, size_t height)
{
    size_t useKernelHeight = height / 2;
    size_t heightResiduals = height % 2;
    size_t useKernelWidth = width / 2;
    size_t widthResiduals = width % 2;
    _word_t *srcPtrs[2];
    _word_t *dstPtrs[2];
    size_t srcLoc, dstLoc;
    // 先ず2x2を適用できる部分を転置する
    for (size_t i = 0; i < useKernelHeight * width; i += 2 * width)
    {
        for (size_t j = 0; j < useKernelWidth; j += 2)
        {
            srcLoc = width * i + j;
            dstLoc = height * j + i;
            srcPtrs[0] = &src[srcLoc]; // TODO: この表記で問題ないか確認する
            srcPtrs[1] = &src[srcLoc + width];
            dstPtrs[0] = &dst[dstLoc];
            dstPtrs[1] = &dst[dstLoc + height];
            _WORD2x2TransposeKernel(srcPtrs, dstPtrs);
        }
        // 残り1列を転置する
        if (widthResiduals == 1)
        {
            srcLoc = width * i + useKernelWidth;
            dstLoc = height * useKernelWidth + i;
            dst[dstLoc] = src[srcLoc];
            dst[dstLoc + height] = src[srcLoc + width];
        }
    }
    // 残り一行を転置する
    if (heightResiduals == 1)
    {
        for (size_t j = 0; j < width; j++)
        {
            srcLoc = width * useKernelHeight + j;
            dstLoc = height * j + useKernelHeight;
            dst[dstLoc] = src[srcLoc];
        }
    }
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

inline void HWORD4x4TransposeKernel(hword_t **inputPtrList /* 元小行列の先頭ポインタ4個のリスト */,
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

inline void CHAR8x8TransposeKernel(char_t **inputPtrList /* 元小行列の先頭ポインタ8個のリスト */,
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

void WORD4x4Transpose(hword_t *src, hword_t *dst, size_t width, size_t height)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            dst[y * width + x] = src[x * height + y];
        }
    }
}
void CHAR8x8Transpose(char_t *src, char_t *dst, size_t width, size_t height)
{
    for (size_t y = 0; y < height; ++y)
    {
        for (size_t x = 0; x < width; ++x)
        {
            dst[y * width + x] = src[x * height + y];
        }
    }
}