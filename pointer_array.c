#include <stdio.h>
#include <stdlib.h>

void main() {
  void *pointerArray;
  pointerArray = (void *)malloc(sizeof(int) * 10);
  printf("pointerArray=%16x\t&pointerArray=%16x\n", pointerArray,
         &pointerArray);

  for (int i = 0; i < 10; i++)
    printf("pointerArray[%d]: 0x_val=%16x\n", i, &pointerArray[i]);

  int *input = (int *)pointerArray;
  for (int i = 0; i < 10; i++)
    printf("input[%d]:        0x_val=%16x\n", i, &input[i]);

  for (int i = 0; i < 10; i++) {
    input[i] = i;
    printf("input[%d]:        val=%d\n", i, input[i]);
  }

  int *out = (int *)pointerArray;
  for (int i = 0; i < 10; i++) printf("pointerArray[%d]: val=%d\n", i, out[i]);

  free(pointerArray);
}
