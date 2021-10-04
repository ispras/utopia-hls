# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.0)
project(UtopiaTest VERSION 0.0)

set(TEST_TARGET utest)
set(CMAKE_CXX_STANDARD 17)

# Check if the 'gtest' target has been already imported (e.g., by LLVM).
if(NOT TARGET gtest)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
  )

  # For Windows: Prevent overriding the parent project's compiler/linker settings.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

add_executable(${TEST_TARGET}
  sample_test.cpp
)

target_link_libraries(${TEST_TARGET} PRIVATE gtest_main)

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET})
