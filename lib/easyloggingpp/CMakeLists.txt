# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 2.8.7)

set(ELPP_LIBRARY easyloggingpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(${ELPP_LIBRARY} easylogging++.cc)
target_include_directories(${ELPP_LIBRARY} PUBLIC .)
target_compile_options(${ELPP_LIBRARY}
  PRIVATE
    -Wno-range-loop-construct
)
