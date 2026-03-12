#include <iostream>
#include "Matrix.h"
#include <fstream>
#include <random>
#include <chrono>
#include <omp.h>

template<typename T>
/* Рекурсивное вычисление определителя через миноры матрицы. Сложность: O(n!) */
T naive_determinant(const Matrix<T>& m) {
    if (m.rows() != m.columns()) {
        throw std::logic_error("Матрица не является квадратной");
    }
    if (m.rows() == 1) return m(0, 0);
    if (m.rows() == 2) return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

    T result = T();
    Matrix<T> temp;

    #pragma omp parallel for shared(m) reduction(+:result) private(temp)
    for (long v = 0; v < m.rows(); v++) {

        temp = Matrix<T>(m.rows() - 1, m.rows() - 1);

        for (size_t i = 1; i < m.rows(); i++) {
            size_t z = 0;
            for (size_t j = 0; j < m.rows(); j++) {
                if (j != v) {
                    temp(i-1, z) = m(i, j);
                    z++;
                }
            }
        }
        if (v % 2 == 0)
            result += (m(0, v) * naive_determinant(temp));
        else
            result += (-m(0, v) * naive_determinant(temp));
    }

    return result;
}


template<typename T>
/* Вычисление определителя через LU-разложение матрицы. Сложность: O(n^3) */
T lu_determinant(const Matrix<T>& m) {
    if (m.rows() != m.columns()) {
        throw std::logic_error("Матрица не является квадратной");
    }

    Matrix<T> l = Matrix<T>(m.rows(), m.columns());
    Matrix<T> u = Matrix<T>(m.rows(), m.columns());

    #pragma omp parallel for shared(l)
    for (long i = 0; i < m.rows(); i++) {
        l(i, i) = T(1);
    }


    T sum = T();
    for (long i = 0; i < m.rows(); i++) {
        for (long j = 0; j < m.rows(); j++) {
            sum = T();
            if (i <= j) {
                #pragma omp parallel for shared(u, l, m) reduction(+:sum)
                for (long k = 0; k < i; k++) {
                    sum += l(i, k) * u(k, j);
                }
                u(i, j) = m(i, j) - sum;
            }
            else {
                #pragma omp parallel for shared(u, l, m) reduction(+:sum)
                for (long k = 0; k < j; k++) {
                    sum += l(i, k) * u(k, j);
                }
                l(i, j) = (m(i, j) - sum) / u(j, j);
            }
        }
    }

    T det = 1;
    #pragma omp parallel for shared(u, l) reduction(*:det)
    for (long i = 0; i < l.rows(); i++) { 
        det *= (l(i, i) * u(i, i));
    }
    return det;
}

Matrix<double> random_square_matrix(size_t n) {
    std::random_device rd = std::random_device();
    std::mt19937 randint(rd());
    std::uniform_real_distribution<double> dist(-0.1, 0.1); // Чтобы значения определителя не улетали в inf

    Matrix<double> m(n, n);
    for (size_t i = 0; i < m.rows(); i++) {
        for (size_t j = 0; j < m.rows(); j++) {
            m(i, j) = dist(randint);
        }
    }
    return m;
}

double benchmark_lu(const Matrix<double>& matrix) {
    std::cout << "lu_determinant() [size:" << matrix.rows() << "x" << matrix.rows() << "]\n";
    auto start = std::chrono::steady_clock::now();
    double result = lu_determinant(matrix);
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ns = duration_cast<std::chrono::nanoseconds>(end - start).count();
    double elapsed_ms = (double) elapsed_ns / 1000000;
    std::cout << "result: " << result << "\n";
    std::cout << "time elapsed: " << elapsed_ms << "ms\n";
    return result;
}

double benchmark_naive(const Matrix<double>& matrix) {
    std::cout << "naive_determinant() [size:" << matrix.rows() << "x" << matrix.rows() << "]\n";
    auto start = std::chrono::steady_clock::now();
    double result = naive_determinant(matrix);
    auto end = std::chrono::steady_clock::now();

    auto elapsed_ns = duration_cast<std::chrono::nanoseconds>(end - start).count();
    double elapsed_ms = (double) elapsed_ns / 1000000;
    std::cout << "result: " << result << "\n";
    std::cout << "time elapsed: " << elapsed_ms << "ms\n";
    return result;
}

int main() {
    omp_set_num_threads(16);
    size_t n, m = 0;
    std::cin >> n >> m;

    Matrix<double> matrix = Matrix<double>(n, m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            std::cin >> matrix(i, j);
        }
    }

    // std::cout << matrix << matrix.rows() << matrix.columns() << "\n";
    std::cout << "threads: " << omp_get_max_threads() << "\n";
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    benchmark_lu(matrix);
    //benchmark_naive(matrix);
    return 0;
}