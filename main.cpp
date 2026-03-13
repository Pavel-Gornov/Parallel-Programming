#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    std::cout << "Hello, MPI!\n";

    int version, subversion;
    MPI_Get_version(&version, &subversion);
    std::cout << "MPI Version: "<< version << "." << subversion << "\n";    
    return 0;
}
