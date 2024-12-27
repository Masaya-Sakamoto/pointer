#include "types.h"
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <string.h>

static const int CIRCULAR_BUFFER_SIZE = 1024;
static const int BLOCK_DIMS = 128;
static const int CONTINUOUS_LINE_LENGTH = 2 * BLOCK_DIMS - 1;
static const int MINI_BLK_DIMS = 8;

// 開始地点をsrc,
void blkCpy_avx512(c16_t *src, c16_t *dst, int forward_skips, int backward_skips)
{
}

void blkCpy(c16_t *src, c16_t *dst)
{
}

void nonoptim_subroutine_blkCpy(c16_t *src, c16_t *dst, int forward_skips, int backward_skips)
{
    // initialize tmp arrays
    c16_t *m0, *m1, *m2, *m3, *m4, *m5, *m6, *m7;
    m0 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m1 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m2 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m3 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m4 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m5 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m6 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));
    m7 = (c16_t *)aligned_alloc(64, MINI_BLK_DIMS * sizeof(c16_t));

    // memcpy src -> tmp
    // for (int skip = 0; skip < CONTINUOUS_LINE_LENGTH; skip+=MINI_BLK_DIMS*2-1)
}

void nonoptim_blkCpy(c16_t *src, c16_t *dst)
{
}

void myMemCpy(c16_t *src, c16_t *dst)
{
}

void nonoptim_myMemCpy(c16_t *src, c16_t *dst, int start_point)
{
    int circ_start = (start_point + CIRCULAR_BUFFER_SIZE) % CIRCULAR_BUFFER_SIZE;
    int circ_end = (start_point + CONTINUOUS_LINE_LENGTH + CIRCULAR_BUFFER_SIZE) % CIRCULAR_BUFFER_SIZE;
    if (circ_start < circ_end)
    {
        memcpy(&dst[0], &src[start_point], CONTINUOUS_LINE_LENGTH * sizeof(c16_t));
    }
    else
    {
        memcpy(&dst[0], &src[start_point], (CIRCULAR_BUFFER_SIZE - circ_start) * sizeof(c16_t));
        memcpy(&dst[CIRCULAR_BUFFER_SIZE - circ_start], &src[0], circ_end * sizeof(c16_t));
    }
}

int main()
{
    c16_t *circ_buffer; //
    c16_t *srcArray;    //
    c16_t *dstArray;    //
    int start = 0;
    int circ_start, circ_end;

    // memory allocation
    circ_buffer = (c16_t *)aligned_alloc(64, CIRCULAR_BUFFER_SIZE * sizeof(c16_t));
    srcArray = (c16_t *)aligned_alloc(64, CONTINUOUS_LINE_LENGTH * sizeof(c16_t));
    dstArray = (c16_t *)aligned_alloc(64, BLOCK_DIMS * BLOCK_DIMS * sizeof(c16_t));

    // init circ_buffer
    for (int i = 0; i < CIRCULAR_BUFFER_SIZE; i++)
    {
        circ_buffer[i].r = i;
        circ_buffer[i].i = i;
    }

    // myMemCpyに置き換えたい
    nonoptim_myMemCpy(circ_buffer, srcArray, start);
}