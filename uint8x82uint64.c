#include <stdint.h>
#include <stdio.h>
#define SMALL_ARRAY_SIZE 8

void smallMemCpy(uint8_t *input, uint8_t *output)
{
	uint64_t value = 0;
	uint64_t *ptr = (uint64_t *)input;
	value = *ptr;
	// printf("input val: %016lx\n", value);
	*(uint64_t *)output = value;
}

void smallMemCpyWithMask(uint8_t *input, uint8_t *output, uint64_t mask)
{
	uint64_t value = 0;
	uint64_t *ptr = (uint64_t *)input;
	value = (*ptr) & mask;
	// printf("input val: %016lx\n", value);
	*(uint64_t *)output = value;
}

void smallMemCpyWithMaskAndShift(uint8_t *input, uint8_t *output, uint64_t mask, size_t shift)
{
	uint64_t value = 0;
	uint64_t *ptr = (uint64_t *)input;
	value = ((*ptr) & mask) >> shift;
	// printf("input val: %016lx\n", value);
	*(uint64_t *)output = value;
}

void smallMemCpyWithMaskAndPeriodicShift(uint8_t *input, uint8_t *output)
{
	const uint64_t mask = 0xffffffffffffff00;
	const size_t shift = 8;
	const size_t smallArraySize = 64;
	uint64_t value = 0;
	int8_t tmp = input[0];
	uint64_t *ptr = (uint64_t *)input;
	value = ((*ptr) & mask) >> shift;
	// printf("input val: %016lx\n", value);
	*(uint64_t *)output = value;
	output[7] = tmp;
}

int main(void)
{
	uint8_t input[SMALL_ARRAY_SIZE];
	uint8_t output[SMALL_ARRAY_SIZE];
	for (size_t i = 0; i < SMALL_ARRAY_SIZE; i++)
	{
		input[i] = i+1;
	}
	printf("input:  ");
	for (size_t i = 0; i < SMALL_ARRAY_SIZE; i++)
	{
		printf("%u ", input[i]);
	}
	printf("\n");

	smallMemCpyWithMaskAndPeriodicShift(input, output);

	printf("output: ");
	for (size_t i = 0; i < SMALL_ARRAY_SIZE; i++)
	{
		printf("%u ", output[i]);
	}
	printf("\n");
	return 0;
}
