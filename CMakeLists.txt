# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)

project(Utopia VERSION 0.0)

include(MLIR.cmake)

add_subdirectory(src)
add_subdirectory(test)

# MLIR example from the LLVM project
add_subdirectory(src/hls/model/mlir)
