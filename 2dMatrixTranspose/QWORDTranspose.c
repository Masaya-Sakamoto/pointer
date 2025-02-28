#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define BITS_PER_BYTE 8
#define DWORD_SIZE 8
#define _WORD_SIZE 4
#define HWORD_SIZE 2
#define CHAR_SIZE 1

typedef uint64_t dword_t;
typedef uint32_t _word_t;
typedef uint16_t hword_t;
typedef uint8_t char_t;

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
    // 先ず2x2を適用できる部分を転置する
    for (size_t i = 0; i < useKernelHeight * width; i += 2 * width)
    {
        for (size_t j = 0; j < useKernelWidth; j += 2)
        {
            size_t srcLoc = width * i + j;
            size_t dstLoc = height * j + i;
            srcPtrs[0] = &src[srcLoc]; // ??
            srcPtrs[1] = &src[srcLoc + width];
            dstPtrs[0] = &dst[dstLoc];
            dstPtrs[1] = &dst[dstLoc + height];
            _WORD2x2TransposeKernel(srcPtrs, dstPtrs);
        }
        // 残り1列を転置する
        if (widthResiduals != 0)
        {
            //
        }
    }
}

inline void HWORD4x4TransposeKernel(hword_t **inputPtrList /* 元小行列の先頭ポインタ4個のリスト */,
                                    hword_t **outputPtrList /* 転置先小行列の先頭ポインタ4個のリスト */)
{
}

inline void CHAR8x8TransposeKernel(char_t **inputPtrList /* 元小行列の先頭ポインタ8個のリスト */,
                                   char_t **outputPtrList /* 転置先小行列の先頭ポインタ8個のリスト */)
{
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