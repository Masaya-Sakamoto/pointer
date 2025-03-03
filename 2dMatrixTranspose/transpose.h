#include "Hword.h"
#include "dword.h"
#include "qword.h"

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
void _word2x2Transpose(_word_t *src, _word_t *dst, size_t width, size_t height);