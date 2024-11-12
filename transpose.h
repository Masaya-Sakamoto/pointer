#include <cstddef>

template <typename T>
inline void transpose(size_t width, size_t height, const T *__restrict pSrc, ptrdiff_t srcLineStride,
                      T *__restrict pDst, ptrdiff_t dstLineStride) noexcept;

