#include <iostream>
#include <dlfcn.h>

int main() {
    void* handle = dlopen("./libcuda_add.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << '\n';
        return 1;
    }

    typedef void (*addVectors_t)(int*, const int*, const int*, int);
    addVectors_t addVectors = (addVectors_t)dlsym(handle, "addVectors");
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        std::cerr << "Cannot load symbol 'addVectors': " << dlsym_error << '\n';
        dlclose(handle);
        return 1;
    }

    const int size = 10;
    int a[size] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int b[size] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
    int c[size] = {0};

    addVectors(c, a, b, size);

    for (int i = 0; i < size; i++) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    dlclose(handle);
    return 0;
}

/*
chatgpt作ホスト側メインプログラム
g++ -o main main.cpp -ldl
*/
