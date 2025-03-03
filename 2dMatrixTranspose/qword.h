#include "types.h"
#include <stdio.h>

/*
u32 4x4
u32 4x2
u32 inv4x2
*/

inline void qword4x4TransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ４個のリスト */,
                                    _word_t **outputPtrList /* 転置先小行列の先頭ポインタ４個のリスト */)
{
    v4i_t a0, a1, a2, a3;
    v4i_t b0, b1, b2, b3;

    // load data and store lower and upper 32bits into different registers
    a0 = _mm_loadu_si128((const __m128i *)inputPtrList[0]);
    a1 = _mm_loadu_si128((const __m128i *)inputPtrList[1]);
    b0 = _mm_unpacklo_epi32(a0, a1); // _mm_unpacklo_epi32
    a2 = _mm_loadu_si128((const __m128i *)inputPtrList[2]);
    a3 = _mm_loadu_si128((const __m128i *)inputPtrList[3]);
    b1 = _mm_unpacklo_epi32(a2, a3); //

    // store residual higher parts into different registers
    b2 = _mm_unpackhi_epi32(a0, a1); //
    b3 = _mm_unpackhi_epi32(a2, a3); //

    // TODO: put some comment here
    a0 = _mm_unpacklo_epi64(b0, b1);
    a1 = _mm_unpacklo_epi64(b2, b3);
    a2 = _mm_unpackhi_epi64(b0, b1);
    a3 = _mm_unpackhi_epi64(b2, b3);

    // store the result into outputPtrList
    _mm_storeu_si128((__m128i *)outputPtrList[0], a0);
    _mm_storeu_si128((__m128i *)outputPtrList[1], a1);
    _mm_storeu_si128((__m128i *)outputPtrList[2], a2);
    _mm_storeu_si128((__m128i *)outputPtrList[3], a3);

    return;
}

inline void qword4x2TransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ２個のリスト */,
                                    _word_t **outputPtrList /* 転置先小行列の先頭ポインタ２個のリスト */)
{
    v4i_t a0, a1;
    v4i_t b0, b1;

    // load the input into registers
    a0 = _mm_loadu_si128((__m128i *)inputPtrList[0]);
    a1 = _mm_loadu_si128((__m128i *)inputPtrList[1]);

    // TODO: put some comments here
    b0 = _mm_unpacklo_epi32(a0, a1);
    b1 = _mm_unpackhi_epi32(a0, a1);

    // TODO: put some comments here
    a0 = _mm_unpacklo_epi64(b0, b1);
    a1 = _mm_unpackhi_epi64(b0, b1);

    // store the output from registers
    _mm_storeu_si128((__m128i *)outputPtrList[0], a0);
    _mm_storeu_si128((__m128i *)outputPtrList[1], a1);

    return;
}

inline void qword4x2InverseTransposeKernel(_word_t **inputPtrList /* 元小行列の先頭ポインタ２個のリスト */,
                                           _word_t **outputPtrList /* 転置先小行列の先頭ポインタ２個のリスト */)
{
    v4i_t a0, a1;
    v4i_t b0, b1;

    // load the input into registers
    a0 = _mm_loadu_si128((__m128i *)inputPtrList[0]);
    a1 = _mm_loadu_si128((__m128i *)inputPtrList[1]);

    // TODO: put some comments here
    b0 = _mm_unpacklo_epi64(a0, a1);
    b1 = _mm_unpackhi_epi64(a0, a1);

    // TODO: put some comments here
    a0 = _mm_unpacklo_epi32(b0, b1);
    a1 = _mm_unpackhi_epi32(b0, b1);

    // store the output from registers
    _mm_storeu_si128((__m128i *)outputPtrList[0], a0);
    _mm_storeu_si128((__m128i *)outputPtrList[1], a1);
}
