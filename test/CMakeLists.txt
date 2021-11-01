# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.0)
project(UtopiaTest VERSION 0.0)

set(TEST_TARGET utest)
set(Z3_LIBRARY z3)
set(CMAKE_CXX_STANDARD 17)

# Use custom build of Z3 if Z3_DIR env var is specified
if(DEFINED ENV{Z3_DIR})
  set(Z3_DIR $ENV{Z3_DIR})
  set(ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:$z3_DIR")
  set(Z3_LIB_DIR "${Z3_DIR}/lib")
  set(CMAKE_CXX_FLAGS "-I${Z3_DIR}/include")
  set(CMAKE_EXE_LINKER_FLAGS "-L${Z3_LIB_DIR}")
endif()

set(MAIN_LIBRARY ueda)

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
  hls/debugger/debugger_test.cpp
  hls/parser/hil/hil_test.cpp
  hls/scheduler/scheduler_test.cpp
  rtl/parser/ril/ril_test.cpp
  test_main.cpp
)

target_link_libraries(${TEST_TARGET} PRIVATE gtest_main ${MAIN_LIBRARY} ${Z3_LIBRARY})

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET})
