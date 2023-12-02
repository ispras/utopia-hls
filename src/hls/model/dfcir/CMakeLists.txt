cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(DFCIR LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)


set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
