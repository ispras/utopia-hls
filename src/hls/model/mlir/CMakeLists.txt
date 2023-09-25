cmake_minimum_required(VERSION 3.13.4)
project(hil-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
add_compile_options(-g3 -Wall -Wignored-qualifiers -DYYERROR_VERBOSE)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
# FIXME find out why CIRCT isn't moving AddCIRCT.cmake to ${CIRCT_CMAKE_DIR}
list(APPEND CMAKE_MODULE_PATH "~/circt/cmake/modules")
include(TableGen)
include(AddCIRCT)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CIRCT_INCLUDE_DIRS})

# FIXME resolve circular dependencies w/ model and remove the direct inclusion
include_directories(${Utopia_SOURCE_DIR}/src)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(hil-opt)
add_subdirectory(hil-translate)
