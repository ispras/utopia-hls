## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(DFCIR LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# TODO: Figure out how to pass Tablegen includes the other way.
# Issue #15 (https://github.com/ispras/utopia-hls/issues/15).
include_directories(${MLIR_INCLUDE_DIRS})
find_package(CTemplate REQUIRED COMPONENTS nothreads)
set(TEMPLATES_PATH "${PROJECT_SOURCE_DIR}/templates")
add_subdirectory(include)
add_subdirectory(lib)
