# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

project(Utopia VERSION 0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

## Suppresses "add_custom_command"-warnings with Ninja generator.
cmake_policy(SET CMP0116 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

##===----------------------------------------------------------------------===//
## CMake command line arguments handling

# It is assumed that unless CIRCT & LLVM dependencies are installed into default system paths,
# "CMAKE_PREFIX_PATH" is explicitly provided to specify these dependencies. 

message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

# "INCLUDE_DIRS" is optional and may be omitted.
if(DEFINED INCLUDE_DIRS)
  message(STATUS "Kernel include directories: ${INCLUDE_DIRS}")
endif()

# "SRC_FILES" has to be explicitly provided.
if(DEFINED SRC_FILES)
  message(STATUS "Kernel source files: ${SRC_FILES}")
else()
  message(FATAL_ERROR "SRC_FILES var is not set!")
endif()

# Unless explicit binary name is provided, "umain" is used.
if(NOT DEFINED OUT)
  set(OUT umain)
endif()

# Unless "BUILD_TESTS" is explicitly set to "ON", tests will not be built.
option(BUILD_TESTS OFF)

message(STATUS "Output binary name: ${OUT}")

##===----------------------------------------------------------------------===//

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

find_package(LpSolve REQUIRED)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)
set_property(TARGET LLVMSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS}")
set_property(TARGET MLIRSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}")
set_property(TARGET CIRCTSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CIRCT_INCLUDE_DIRS}")

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_ROOT}/../../../../cmake/modules")

include(TableGen)
include(AddCIRCT)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(src)

if(${BUILD_TESTS} STREQUAL "ON")
  add_subdirectory(test)
  message(STATUS "Building tests: YES")
else()
  message(STATUS "Building tests: NO")
endif()
