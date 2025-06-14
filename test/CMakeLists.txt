## SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

project(UtopiaTest VERSION 0.0)

add_compile_options(-g3 -Wall -Wignored-qualifiers)

## If GTest isn't installed, download sources from repo.
if(NOT TARGET GTest::gtest)
  message(STATUS "Downloading GTest-sources from GitHub")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0)

  ## For Windows: Prevent overriding the parent project's compiler/linker settings.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

enable_testing()

set(EXAMPLES_PATH "${CMAKE_SOURCE_DIR}/examples")
add_subdirectory(examples)
add_subdirectory(model)

add_definitions("-fno-rtti")

add_executable(utest
  ${EXAMPLES_TESTS}
  ${MODEL_TESTS}
  ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cpp
)

## Sets the OS-specific NULLDEVICE to dump SystemVerilog-output into. 
if (WIN32)
  set(NULLDEVICE "NUL")
elseif(UNIX)
  set(NULLDEVICE "/dev/null")
endif()

target_compile_definitions(utest
  PUBLIC
    NULLDEVICE="${NULLDEVICE}"
    TEST_EXAMPLES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples"
    TEST_MODEL_DFCIR_DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/model/dfcir/data"
)

target_include_directories(utest
  PRIVATE
    ${EXAMPLES_PATH}
)

target_link_libraries(utest
  PRIVATE
    GTest::gtest
    MLIRParser
    Utopia::DFCIR
    Utopia::DFCIRPasses
    Utopia::DFCXX
    Utopia::DFCXXSTD
    easyloggingpp
)

include(GoogleTest)
gtest_discover_tests(utest)
