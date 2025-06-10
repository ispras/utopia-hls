## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(DFCXX LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(CTemplate REQUIRED COMPONENTS nothreads)
set(TEMPLATES_PATH "${PROJECT_SOURCE_DIR}/templates")
add_subdirectory(include)
add_subdirectory(lib)

add_subdirectory(std)
