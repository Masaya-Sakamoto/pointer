#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "types.h"

int main() {
    FILE *fp;
    cf_t array[] = {{2,3},{5,7},{11,13},{17,19}};
    fp = fopen("test_1darray_byc.b", "ab");
    if (fp) {
        fwrite(array, sizeof(array), 1, fp);
    }
    fclose(fp);
    return 0;
}