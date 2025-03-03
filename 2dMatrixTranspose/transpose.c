#include "transpose.h"

void _word2x2Transpose(_word_t *src, _word_t *dst, size_t width, size_t height)
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
            _word2x2TransposeKernel(srcPtrs, dstPtrs);
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