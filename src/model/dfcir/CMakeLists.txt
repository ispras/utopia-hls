cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(DFCIR LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# TODO: Figure out how to pass Tablegen includes the other way.
# Issue #15 (https://github.com/ispras/utopia-hls/issues/15).
include_directories(${MLIR_INCLUDE_DIRS}) 
add_subdirectory(include)
add_subdirectory(lib)
