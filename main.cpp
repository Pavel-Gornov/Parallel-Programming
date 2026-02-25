#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Hello, OpenMP!\n";
    std::cout << "version: " << _OPENMP << "\n";
    std::cout << "max threads: " << omp_get_max_threads() << "\n";
    return 0;
}
