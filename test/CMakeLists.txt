# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(UtopiaTest VERSION 0.0)

add_compile_options(-g3 -Wall -Wignored-qualifiers)

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

add_subdirectory(model)

add_executable(utest
  ${MODEL_TESTS}
  test_main.cpp
)

# Sets the OS-specific NULLDEVICE to dump SystemVerilog-output into. 
if (WIN32)
  set(NULLDEVICE "NUL")
elseif(UNIX)
  set(NULLDEVICE "/dev/null")
endif()

target_compile_definitions(utest
  PUBLIC
    NULLDEVICE="${NULLDEVICE}"
)

target_include_directories(utest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(utest
  PRIVATE
    gtest_main
    Utopia::MLIRDFCIR
    Utopia::DFCXX
    easyloggingpp
)

include(GoogleTest)
gtest_discover_tests(utest)
