# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(Utopia VERSION 0.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Check if UTOPIA_HLS_HOME env var is set
if(DEFINED ENV{UTOPIA_HLS_HOME})
  message(STATUS "UTOPIA_HLS_HOME: $ENV{UTOPIA_HLS_HOME}")
else()
  message(FATAL_ERROR "UTOPIA_HLS_HOME env var is not set!")
endif()

find_package(MLIR REQUIRED CONFIG)
set_property(TARGET LLVMSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${LLVM_INCLUDE_DIRS}")
set_property(TARGET MLIRSupport APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MLIR_INCLUDE_DIRS}")

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
