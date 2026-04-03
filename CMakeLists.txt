cmake_minimum_required(VERSION 3.18)
project("Parallel" LANGUAGES CXX CUDA)

set(EXECUTABLE ${PROJECT_NAME})

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
add_executable(${EXECUTABLE} main.cu)