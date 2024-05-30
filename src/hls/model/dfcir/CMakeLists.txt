cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(DFCIR LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include_directories(${MLIR_INCLUDE_DIRS}) # TODO: Figure out how to pass Tablegen includes the other way.
add_subdirectory(include)
add_subdirectory(lib)
