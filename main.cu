#include <iostream>
#include <cstdio>
//#include <cuda/cmath>
#include "Matrix.h"
#include <chrono>

// Kernel function that runs on the GPU
__global__ void kernel_gpu_multiply(double* Ag, double* Bg, double* Cg, int rows, int columns, int A_columns) {
    /* A_columns == B.rows(); rows == C.rows() == A.rows(); columns == B.columns() == C.columns() */
    size_t row = blockIdx.y*blockDim.y+threadIdx.y;
    size_t col = blockIdx.x*blockDim.x+threadIdx.x;

    double t_sum = 0;

    if (row < rows && col < columns) {
        
        for (int r = 0; r < A_columns; r++) {
            t_sum += Ag[row * A_columns + r] * Bg[r * columns + col];
        }
        Cg[row * columns + col] = t_sum;
    }
}

Matrix<double> gpu_multiply(Matrix<double> A, Matrix<double> B) {
    if (A.columns() != B.rows()) {
        throw std::logic_error("Матрицы не совместны");
    }
    double* Ag = nullptr;
    double* Bg = nullptr;
    double* Cg = nullptr;

    Matrix<double> C = Matrix<double>(A.rows(), B.columns());

    cudaError_t result;
    result = cudaMalloc((void**)&Ag, A.rows()*A.columns()*sizeof(double));
    if (result != cudaSuccess) throw std::runtime_error("Ошибка при выделении памяти!");

    result = cudaMemcpy(Ag, A.data(), A.rows()*A.columns()*sizeof(double), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) throw std::runtime_error("Ошибка при копировании памяти!");

    result = cudaMalloc((void**)&Bg, B.rows()*B.columns()*sizeof(double));
    if (result != cudaSuccess) throw std::runtime_error("Ошибка при выделении памяти!");

    result = cudaMemcpy(Bg, B.data(), B.rows()*B.columns()*sizeof(double), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) throw std::runtime_error("Ошибка при копировании памяти!");

    result = cudaMalloc((void**)&Cg, C.rows()*C.columns()*sizeof(double));
    if (result != cudaSuccess) throw std::runtime_error("Ошибка при выделении памяти!");

    size_t N = std::max(std::max(A.rows(), B.rows()), std::max(A.columns(), B.columns()));
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512) {
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    std::cout << "rows: " << C.rows() << " columns: " << C.columns() << " A_columns: " << A.columns() << "\n";
    kernel_gpu_multiply<<<threadsPerBlock, blocksPerGrid>>>(Ag, Bg, Cg, C.rows(), C.columns(), A.columns());
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), Cg, C.rows()*C.columns()*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(Ag);
    cudaFree(Bg);
    cudaFree(Cg);

    return C;
}

int main() {
    #ifdef _WIN32
        system("chcp 65001 > nul");
    #endif
    //std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::cout << "Hello World from the CPU!\n";

    size_t n, m;
    std::cin >> n;
    std::cin >> m;

    Matrix<double> matrix = Matrix<double>(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            std::cin >> matrix(i, j);
        }
    }

    std::cin >> n;
    std::cin >> m;

    Matrix<double> matrix2 = Matrix<double>(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            std::cin >> matrix2(i, j);
        }
    }

    std::cout << "gpu_multiply() [size:" << matrix.rows() << "x" << matrix.rows() << "]\n";

    auto start = std::chrono::steady_clock::now();
    Matrix<double> result = gpu_multiply(matrix, matrix2);
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    double elapsed_ms = (double) elapsed_ns / 1000000;
    std::cout << "time elapsed: " << elapsed_ms << "ms\n";
    std::cout << "result:\n" << result << "\n";
    return 0;
}