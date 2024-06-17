# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.0)

project(UtopiaTest VERSION 0.0)

set(TEST_TARGET utest)

# Karatsuba multiplier needed for some tests.
set(MULGEN_TARGET mulgen)

add_compile_options(-g3 -Wall -Wignored-qualifiers -DYYERROR_VERBOSE)

# Check if the 'gtest' target has been already imported (e.g., by LLVM).
if(NOT TARGET gtest)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        609281088cfefc76f9d0ce82e1ff6c30cc3591e5)

  # For Windows: Prevent overriding the parent project's compiler/linker settings.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

add_executable(${TEST_TARGET}
  #hls/compiler/compiler_dfc_test.cpp
  #hls/compiler/compiler_hil_test.cpp
  #hls/debugger/eq_checker_test.cpp
  #hls/library/parser_test.cpp
  #hls/model/mlir/conversion_dfc_test.cpp
  #hls/model/mlir/dump_test.cpp
  #hls/model/mlir/instance_dfc_test.cpp
  #hls/model/mlir/transform_test.cpp
  #hls/parser/dfc/dfc_dot_test.cpp
  #hls/parser/dfc/dfc_idct_test.cpp
  #hls/parser/dfc/dfc_model_types_test.cpp
  #hls/parser/hil/hil_test.cpp
  #hls/scheduler/param_optimizer_test.cpp
  #hls/scheduler/scheduler_test.cpp
  #hls/scheduler/simulated_annealing_test.cpp
  #lib/minisat/minisat_test.cpp
  hls/model/dfcxx/simple.cpp
  hls/model/dfcxx/scalar3.cpp
  hls/model/dfcxx/matrixmul2.cpp
  hls/model/dfcxx/muxmul.cpp
  hls/model/dfcxx/addconst.cpp
  hls/model/dfcxx/movingsum.cpp
  hls/model/dfcxx/idct.cpp
  test_main.cpp
  #utils/graph_traverse_test.cpp
)

# Sets the OS-specific NULLDEVICE to dump SystemVerilog-output into. 
if (WIN32)
  set(NULLDEVICE "NUL")
elseif(UNIX)
  set(NULLDEVICE "/dev/null")
endif()

target_compile_definitions(${TEST_TARGET}
  PUBLIC
    NULLDEVICE="${NULLDEVICE}"
)

target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TEST_TARGET}
  PRIVATE
    gtest_main
    Utopia::MLIRDFCIR
    Utopia::DFCXX
    easyloggingpp
    minisat-lib-static
)

#add_executable(${MULGEN_TARGET}
#  hls/library/mulgen.cpp
#)
#target_link_libraries(${MULGEN_TARGET}
#  PRIVATE
#    stdc++fs
#)

include(GoogleTest)
gtest_discover_tests(${TEST_TARGET})

file(COPY data DESTINATION .)
