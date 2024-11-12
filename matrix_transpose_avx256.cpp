#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

// original https://qiita.com/beru/items/12b4249c95a012a28ccd
template <typename T>
inline void transpose(size_t width, size_t height, const T *__restrict pSrc, ptrdiff_t srcLineStride,
                      T *__restrict pDst, ptrdiff_t dstLineStride) noexcept
{
    if (width == 0 || height == 0 || pSrc == nullptr || pDst == nullptr || srcLineStride < width ||
        dstLineStride < height)
    {
        return;
    }

    if (width >= 16 / sizeof(T) && height >= 16)
    {
        size_t bw = std::min<size_t>(128u / sizeof(T), flp2(width));
        size_t bh = std::min<size_t>(512u / sizeof(T), flp2(height));
        size_t nw = width / bw;
        size_t nh = height / bh;
        for (size_t i = 0; i < nh; ++i)
        {
            for (size_t j = 0; j < nw; ++j)
            {
                auto pSrc2 = &pSrc[i * bh * srcLineStride + j * bw];
                auto pDst2 = &pDst[j * bw * dstLineStride + i * bh];
                constexpr size_t n = 16;
                for (size_t l = 0; l < bh / n; ++l)
                {
                    for (size_t k = 0; k < bw / n; ++k)
                    {
                        auto pSrc3 = &pSrc2[l * n * srcLineStride + k * n];
                        auto pDst3 = &pDst2[k * n * dstLineStride + l * n];
                        transpose_1b_16x16(pSrc3, srcLineStride, pDst3, dstLineStride);
                    }
                }
            }
        }
        size_t widthRemain = width % bw;
        size_t heightRemain = height % bh;
        size_t width2 = bw * nw;
        size_t height2 = bh * nh;
        if (widthRemain)
        {
            transpose(widthRemain, height2, pSrc + width2, srcLineStride, pDst + (width2 * dstLineStride),
                      dstLineStride);
        }
        if (heightRemain)
        {
            transpose(width, heightRemain, pSrc + height2 * srcLineStride, srcLineStride, pDst + height2,
                      dstLineStride);
        }
    }
    else
    {
        transpose_naive(width, height, pSrc, srcLineStride, pDst, dstLineStride);
    }
}

// https://stackoverflow.com/a/2681094
inline uint32_t flp2(uint32_t x)
{
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x - (x >> 1);
}

// original https://qiita.com/beru/items/12b4249c95a012a28ccd
inline void transpose_1b_16x16(const uint8_t *__restrict pSrc, ptrdiff_t srcLineStride, uint8_t *__restrict pDst,
                               ptrdiff_t dstLineStride) noexcept
{
    // ２列配列を取り出す
    __m256i a0 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 8 * srcLineStride), (__m128i const *)(pSrc + 0 * srcLineStride));
    __m256i a1 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 9 * srcLineStride), (__m128i const *)(pSrc + 1 * srcLineStride));
    __m256i a2 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 10 * srcLineStride), (__m128i const *)(pSrc + 2 * srcLineStride));
    __m256i a3 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 11 * srcLineStride), (__m128i const *)(pSrc + 3 * srcLineStride));
    __m256i a4 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 12 * srcLineStride), (__m128i const *)(pSrc + 4 * srcLineStride));
    __m256i a5 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 13 * srcLineStride), (__m128i const *)(pSrc + 5 * srcLineStride));
    __m256i a6 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 14 * srcLineStride), (__m128i const *)(pSrc + 6 * srcLineStride));
    __m256i a7 =
        _mm256_loadu2_m128i((__m128i const *)(pSrc + 15 * srcLineStride), (__m128i const *)(pSrc + 7 * srcLineStride));

    __m256i b0 = _mm256_unpacklo_epi8(a0, a1); // latency: 1.0, throughput: 0.5
    __m256i b1 = _mm256_unpacklo_epi8(a2, a3);
    __m256i b2 = _mm256_unpacklo_epi8(a4, a5);
    __m256i b3 = _mm256_unpacklo_epi8(a6, a7);
    __m256i b4 = _mm256_unpackhi_epi8(a0, a1);
    __m256i b5 = _mm256_unpackhi_epi8(a2, a3);
    __m256i b6 = _mm256_unpackhi_epi8(a4, a5);
    __m256i b7 = _mm256_unpackhi_epi8(a6, a7);

    a0 = _mm256_unpacklo_epi16(b0, b1); // latency: 1.0, throughput: 0.5
    a1 = _mm256_unpacklo_epi16(b2, b3);
    a2 = _mm256_unpackhi_epi16(b0, b1);
    a3 = _mm256_unpackhi_epi16(b2, b3);
    a4 = _mm256_unpacklo_epi16(b4, b5);
    a5 = _mm256_unpacklo_epi16(b6, b7);
    a6 = _mm256_unpackhi_epi16(b4, b5);
    a7 = _mm256_unpackhi_epi16(b6, b7);

    b0 = _mm256_unpacklo_epi32(a0, a1); // latency: 1.0, throughput: 0.5
    b1 = _mm256_unpackhi_epi32(a0, a1);
    b2 = _mm256_unpacklo_epi32(a2, a3);
    b3 = _mm256_unpackhi_epi32(a2, a3);
    b4 = _mm256_unpacklo_epi32(a4, a5);
    b5 = _mm256_unpackhi_epi32(a4, a5);
    b6 = _mm256_unpacklo_epi32(a6, a7);
    b7 = _mm256_unpackhi_epi32(a6, a7);

    a0 = _mm256_permute4x64_epi64(b0, _MM_SHUFFLE(3, 1, 2, 0)); // latency: 3.0, throughput: 1.0
    a1 = _mm256_permute4x64_epi64(b1, _MM_SHUFFLE(3, 1, 2, 0));
    a2 = _mm256_permute4x64_epi64(b2, _MM_SHUFFLE(3, 1, 2, 0));
    a3 = _mm256_permute4x64_epi64(b3, _MM_SHUFFLE(3, 1, 2, 0));
    a4 = _mm256_permute4x64_epi64(b4, _MM_SHUFFLE(3, 1, 2, 0));
    a5 = _mm256_permute4x64_epi64(b5, _MM_SHUFFLE(3, 1, 2, 0));
    a6 = _mm256_permute4x64_epi64(b6, _MM_SHUFFLE(3, 1, 2, 0));
    a7 = _mm256_permute4x64_epi64(b7, _MM_SHUFFLE(3, 1, 2, 0));
    _mm256_storeu2_m128i((__m128i *)(pDst + 1 * dstLineStride), (__m128i *)(pDst + 0 * dstLineStride), a0);
    _mm256_storeu2_m128i((__m128i *)(pDst + 3 * dstLineStride), (__m128i *)(pDst + 2 * dstLineStride), a1);
    _mm256_storeu2_m128i((__m128i *)(pDst + 5 * dstLineStride), (__m128i *)(pDst + 4 * dstLineStride), a2);
    _mm256_storeu2_m128i((__m128i *)(pDst + 7 * dstLineStride), (__m128i *)(pDst + 6 * dstLineStride), a3);
    _mm256_storeu2_m128i((__m128i *)(pDst + 9 * dstLineStride), (__m128i *)(pDst + 8 * dstLineStride), a4);
    _mm256_storeu2_m128i((__m128i *)(pDst + 11 * dstLineStride), (__m128i *)(pDst + 10 * dstLineStride), a5);
    _mm256_storeu2_m128i((__m128i *)(pDst + 13 * dstLineStride), (__m128i *)(pDst + 12 * dstLineStride), a6);
    _mm256_storeu2_m128i((__m128i *)(pDst + 15 * dstLineStride), (__m128i *)(pDst + 14 * dstLineStride), a7);
}
