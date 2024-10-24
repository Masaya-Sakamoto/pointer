#include <stdio.h>
#include <stdlib.h>
#include "types.h"

int main () {
    cf_t *buffer;
    cf_t ***store;
    buffer = (cf_t *)malloc(sizeof(cf_t) * 608);
    store = (cf_t **)malloc
    FILE *fp;
    fp = fopen("sampledata3.b", "rb");
    fread(buffer, sizeof(cf_t)*4, 1, fp);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("(%f, %f), ", buffer[i][j].r, buffer[i][j].i);
        }
    }
    printf("\n");
    return 0;
}