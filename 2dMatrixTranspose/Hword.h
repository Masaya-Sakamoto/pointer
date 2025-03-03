#include "types.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
u32 8x8
u32 8x4
u32 inv8x4
u32 8x2
u32 inv8x2
*/

inline __m256i _mm256_unpacklo_epi128(__m256i a, __m256i b)
{
    __m128i a_lo = _mm256_extracti128_si256(a, 0); // a の前半 128 ビット
    __m128i b_lo = _mm256_extracti128_si256(b, 0); // b の前半 128 ビット
    return _mm256_set_m128i(b_lo, a_lo);           // b_lo を上位、a_lo を下位に配置
}

inline __m256i _mm256_unpackhi_epi128(__m256i a, __m256i b)
{
    __m128i a_hi = _mm256_extracti128_si256(a, 1); // a の後半 128 ビット
    __m128i b_hi = _mm256_extracti128_si256(b, 1); // b の後半 128 ビット
    return _mm256_set_m128i(b_hi, a_hi);           // b_hi を上位、a_hi を下位に配置
}

inline void Hword_8x8TransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ８個のリスト */,
                                     _word_t **outputPtrList /* 転置先小行列の先頭ポインタ８個のリスト */)
{
    v8i_t a0, a1, a2, a3, a4, a5, a6, a7;
    v8i_t b0, b1, b2, b3, b4, b5, b6, b7;

    // load data and store lower 32bits into different registers
    a0 = _mm256_loadu_si256((__m256i *)inputPtrList[0]);
    a1 = _mm256_loadu_si256((__m256i *)inputPtrList[1]);
    b0 = _mm256_unpacklo_epi32(a0, a1);
    a2 = _mm256_loadu_si256((__m256i *)inputPtrList[2]);
    a3 = _mm256_loadu_si256((__m256i *)inputPtrList[3]);
    b1 = _mm256_unpacklo_epi32(a2, a3);
    a4 = _mm256_loadu_si256((__m256i *)inputPtrList[4]);
    a5 = _mm256_loadu_si256((__m256i *)inputPtrList[5]);
    b2 = _mm256_unpacklo_epi32(a4, a5);
    a6 = _mm256_loadu_si256((__m256i *)inputPtrList[6]);
    a7 = _mm256_loadu_si256((__m256i *)inputPtrList[7]);
    b3 = _mm256_unpacklo_epi32(a6, a7);

    // store residual higher 32bit into different registers
    b4 = _mm256_unpackhi_epi32(a0, a1);
    b5 = _mm256_unpackhi_epi32(a2, a3);
    b6 = _mm256_unpackhi_epi32(a4, a5);
    b7 = _mm256_unpackhi_epi32(a6, a7);

    // TODO: put some comments here
    a0 = _mm256_unpacklo_epi64(b0, b1); // 0lo 4lo
    a1 = _mm256_unpacklo_epi64(b2, b3); // 0hi 4hi
    a2 = _mm256_unpacklo_epi64(b4, b5); // 1lo 5lo
    a3 = _mm256_unpacklo_epi64(b6, b7); // 1hi 5hi
    a4 = _mm256_unpackhi_epi64(b0, b1); // 2lo 6lo
    a5 = _mm256_unpackhi_epi64(b2, b3); // 2hi 6hi
    a6 = _mm256_unpackhi_epi64(b4, b5); // 3lo 7lo
    a7 = _mm256_unpackhi_epi64(b6, b7); // 3hi 7hi

    // TODO: put some comments here
    b0 = _mm256_permute2x128_si256(a0, a1, 0x20);
    b1 = _mm256_permute2x128_si256(a2, a3, 0x20);
    b2 = _mm256_permute2x128_si256(a4, a5, 0x20);
    b3 = _mm256_permute2x128_si256(a6, a7, 0x20);
    b4 = _mm256_permute2x128_si256(a0, a1, 0x31);
    b5 = _mm256_permute2x128_si256(a2, a3, 0x31);
    b6 = _mm256_permute2x128_si256(a4, a5, 0x31);
    b7 = _mm256_permute2x128_si256(a6, a7, 0x31);

    // store the result to outputPtrList
    _mm256_storeu_si256((__m256i *)outputPtrList[0], b0);
    _mm256_storeu_si256((__m256i *)outputPtrList[1], b1);
    _mm256_storeu_si256((__m256i *)outputPtrList[2], b2);
    _mm256_storeu_si256((__m256i *)outputPtrList[3], b3);
    _mm256_storeu_si256((__m256i *)outputPtrList[4], b4);
    _mm256_storeu_si256((__m256i *)outputPtrList[5], b5);
    _mm256_storeu_si256((__m256i *)outputPtrList[6], b6);
    _mm256_storeu_si256((__m256i *)outputPtrList[7], b7);

    return;
}

inline void Hword_8x4TransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ４個のリスト */,
                                     _word_t **outputPtrList /* 転置先小行列の先頭ポインタ４個のリスト */)
{
    // TODO: implement the transpose kernel for 8x4 matrix using AVX2 instructions
}

inline void Hword_8x4InverseTransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ４個のリスト */,
                                            _word_t **outputPtrList /* 転置先小行列の先頭ポインタ４個のリスト */)
{
    // TODO: implement the inverse transpose kernel for 8x4 matrix using AVX2 instructions
}