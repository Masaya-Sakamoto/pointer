#include "types.h"
#include <cstdlib>
#include <iostream>
// #include <fstream>

static c16_t *intArray;
static cf_t *floatArray;
static const int N = 1024;

int main()
{
    // Initialize the integer array with some values
    intArray = (c16_t *)aligned_alloc(64, sizeof(c16_t) * N);
    floatArray = (cf_t *)aligned_alloc(64, sizeof(cf_t) * N);
    int i;

    for (i = 0; i < N; i++)
    {
        intArray[i].r = i;
        intArray[i].i = i * i;
    }
    // FILE *fp;

    // test int16 -> float mnemonic
    for (i = 0; i < N; i++)
    {
        floatArray[i].r = (float)intArray[i].r;
        floatArray[i].i = (float)intArray[i].i;
    }

    for (i = 0; i < N; i++)
    {
        floatArray[i].r *= 1.1;
        floatArray[i].i *= 2.1;
    }

    for (i = 0; i < N; i++)
    {
        std::cout << floatArray[i].r << "+" << floatArray[i].i << "j" << std::endl;
    }
    return 0;
}