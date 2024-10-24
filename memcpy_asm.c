#include <stdlib.h>
#include <string.h>
#define N 10

int main () {
    int ans = 0;
    int a[N];
    for (int i = 0; i < N; i++) a[i] = i*i + i+2 + 1;
    int b[N];
    memcpy(b, a, sizeof(int) * N);
    for (int i = 1; i < N; i++) ans += b[i] * b[i-1];
    return 0;
}