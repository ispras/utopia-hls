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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()

add_executable(${TEST_TARGET}
  gate/checker/checker_test.cpp
  gate/model/netlist_test.cpp
  hls/compiler/compiler_test.cpp
#  hls/compiler/compiler_dfc_test.cpp
  hls/debugger/debugger_test.cpp
  hls/library/library_test.cpp
  hls/model/mlir/transform_test.cpp
  hls/parser/dfc/dfc_dot_test.cpp
  hls/parser/dfc/dfc_idct_test.cpp
  hls/parser/hil/hil_test.cpp
  hls/scheduler/param_optimizer_test.cpp
  hls/scheduler/scheduler_test.cpp
  rtl/parser/ril/ril_test.cpp
  test_main.cpp
)

target_link_libraries(${TEST_TARGET} PRIVATE gtest_main ${MAIN_LIBRARY})

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET})
