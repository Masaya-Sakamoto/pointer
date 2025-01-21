#include <stdio.h>

int gval = 10;

void setptr(int **ptr)
{
    *ptr = &gval;
}

int main()
{
    // int val = 20;
    // int *ptr = &val;
    int *ptr = NULL;

    // printf("Before setptr: ptr = %p, *ptr = %d\n", (void *)ptr, *ptr);

    setptr(&ptr);

    printf("After setptr: ptr = %p, *ptr = %d\n", (void *)ptr, *ptr);

    return 0;
}
