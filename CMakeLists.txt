# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(Utopia VERSION 0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Check if CIRCT build directory var is set.
if(DEFINED CIRCT_BIN_HOME)
  message(STATUS "CIRCT_BIN_HOME: ${CIRCT_BIN_HOME}")
  list(INSERT CMAKE_PREFIX_PATH 0 "${CIRCT_BIN_HOME}")
endif()

# Check if LLVM build directory var is set.
if(DEFINED LLVM_BIN_HOME)
  message(STATUS "LLVM_BIN_HOME: ${LLVM_BIN_HOME}")
  list(INSERT CMAKE_PREFIX_PATH 0 "${LLVM_BIN_HOME}")
endif()

# Check if UTOPIA_HLS_HOME env var is set.
if(DEFINED ENV{UTOPIA_HLS_HOME})
  message(STATUS "UTOPIA_HLS_HOME: $ENV{UTOPIA_HLS_HOME}")
else()
  message(FATAL_ERROR "UTOPIA_HLS_HOME env var is not set!")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set_property(TARGET LLVMSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS}")
set_property(TARGET MLIRSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include(UtopiaFindHelper)
set_local_from_env(CTemplate_ROOT CT_DIR)
set_local_from_env(XercesC_ROOT XERCES_DIR)
set_local_from_env(Z3_ROOT Z3_DIR)

find_package(LpSolve REQUIRED)
find_package(Z3 REQUIRED)
find_package(XercesC REQUIRED)
find_package(CTemplate REQUIRED COMPONENTS nothreads)

add_subdirectory(lib)
add_subdirectory(src)

add_subdirectory(test)
