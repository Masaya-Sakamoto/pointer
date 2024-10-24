#include <stdio.h>
#include <stdint.h>

int
main()
{
  uintptr_t addr;
  int n = 16;
  int __attribute__ ((aligned(64))) a[n];
  addr = (uintptr_t)a;
  printf("a = %p\n", (void *)a);
  printf("%p %% 64 == %d\n", (void *)a, addr % 64);
  return 0;
}