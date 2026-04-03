#include <iostream>
#include <cstdio>
#include "Matrix.h"

// Kernel function that runs on the GPU
__global__ void helloFromGPU() {
    printf("Hello World from the GPU!\n");
}

int main() {
    std::cout << "Hello World from the CPU!\n";

    // <<<number_of_blocks, threads_per_block>>>
    helloFromGPU<<<1, 1>>>();
    Matrix<int> m = Matrix<int>(2, 2);
    m(0, 0) = 1; m(1, 1) = 1;
    std::cout << m << "\n";
    cudaDeviceSynchronize();
    return 0;
}