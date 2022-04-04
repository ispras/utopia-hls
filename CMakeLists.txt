# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(Utopia VERSION 0.0)

include(MLIR.cmake)

set(MAIN_LIBRARY ueda)          # Used by {src,test}/CMakeLists.txt
set(ELPP_LIBRARY easyloggingpp) # Used by {lib,src}/CMakeLists.txt

# Use custom build of Z3 if Z3_DIR env var is specified
if(DEFINED ENV{Z3_DIR})
  message(STATUS "Z3_DIR env var is set.")
  set(Z3_DIR $ENV{Z3_DIR})
  link_directories(${Z3_DIR}/lib)
  set(CMAKE_CXX_FLAGS "-I${Z3_DIR}/include")
else()
  message(STATUS "Z3_DIR env var is not set.")
endif()

# Use custom build of CTemplate if CT_DIR env var is specified
if(DEFINED ENV{CT_DIR})
  message(STATUS "CT_DIR env var is set.")
  set(CT_DIR $ENV{CT_DIR})
  link_directories(${CT_DIR}/lib)
  set(CMAKE_CXX_FLAGS "-I${CT_DIR}/include ${CMAKE_CXX_FLAGS}")
else()
  message(STATUS "CT_DIR env var is not set.")
endif()

include_directories(lib/easyloggingpp)
include_directories(lib/minisat)

link_directories(${CMAKE_CURRENT_BINARY_DIR}/lib/minisat)

add_subdirectory(lib)
add_subdirectory(src)

add_subdirectory(test)
