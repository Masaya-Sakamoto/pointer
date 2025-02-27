#include <stdint.h>
#include <stdio.h>
#define SMALL_ARRAY_SIZE 8
#define BITS_PER_BYTE 8

void QWORDMemcpy(uint8_t *input, uint8_t *output)
{
    uint64_t value = 0;
    uint64_t *ptr = (uint64_t *)input;
    value = *ptr;
    // printf("input val: %016lx\n", value);
    *(uint64_t *)output = value;
}

void QWORDMemcpyWithMask(uint8_t *input, uint8_t *output, uint64_t mask)
{
    uint64_t value = 0;
    uint64_t *ptr = (uint64_t *)input;
    value = (*ptr) & mask;
    // printf("input val: %016lx\n", value);
    *(uint64_t *)output = value;
}

void QWORDMemcpyWithMaskAndShift(uint8_t *input, uint8_t *output, uint64_t mask, size_t shift)
{
    uint64_t value = 0;
    uint64_t *ptr = (uint64_t *)input;
    value = ((*ptr) & mask) >> shift;
    *(uint64_t *)output = value;
}

void QWORDMemcpyWithPeriodic1Shift(uint8_t *input, uint8_t *output)
{
    const uint64_t mask = 0xffffffffffffff00;
    const size_t shift = 8;
    const size_t smallArraySize = 64;
    uint64_t value = 0;
    int8_t tmp = input[0];
    uint64_t *ptr = (uint64_t *)input;
    value = ((*ptr) & mask) >> shift;
    *(uint64_t *)output = value;
    output[7] = tmp;
}

void QWORDMemcpyWithPeriodicShift(uint8_t *input, uint8_t *output, size_t byte_shift)
{
    const size_t qwordArraySize = sizeof(uint64_t) * BITS_PER_BYTE;
    const uint8_t shift = ((byte_shift + sizeof(uint64_t)) % sizeof(uint64_t)) * BITS_PER_BYTE;
    const uint64_t mask = UINT64_MAX << shift;
    const uint64_t tmp_mask = ~mask;

    uint64_t value = *(uint64_t *)input;
    uint64_t tmp = (value & tmp_mask) << (qwordArraySize - shift);
    value = (value & mask) >> shift;
    value |= tmp;
    *(uint64_t *)output = value;
}

int main(void)
{
    uint8_t input[SMALL_ARRAY_SIZE];
    uint8_t output[SMALL_ARRAY_SIZE];
    for (size_t i = 0; i < SMALL_ARRAY_SIZE; i++)
    {
        input[i] = 240 + i + 1;
    }
    printf("input:  ");
    for (size_t i = 0; i < SMALL_ARRAY_SIZE; i++)
    {
        printf("0x%02X ", input[i]);
    }
    printf("\n");

    QWORDMemcpyWithPeriodicShift(input, output, 3);

    printf("output: ");
    for (size_t i = 0; i < SMALL_ARRAY_SIZE; i++)
    {
        printf("0x%02X ", output[i]);
    }
    printf("\n");
    return 0;
}
