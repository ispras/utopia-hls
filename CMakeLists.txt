# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(Utopia VERSION 0.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(LpSolve REQUIRED)
include(cmake/MLIR.cmake)

set(MAIN_LIBRARY ueda)          # Used by {src,test}/CMakeLists.txt
set(ELPP_LIBRARY easyloggingpp) # Used by {lib,src}/CMakeLists.txt

# Use custom build of Z3 if Z3_DIR env var is specified
if(DEFINED ENV{Z3_DIR})
  message(STATUS "Z3_DIR env var is set.")
  set(Z3_DIR $ENV{Z3_DIR})
  include_directories(SYSTEM ${Z3_DIR}/include)
  link_directories(SYSTEM ${Z3_DIR}/lib)
else()
  find_file(Z3 z3 REQUIRED DOC "Z3 installed to system paths")
  if (Z3)
    message(STATUS "Z3 found: ${Z3}")
  else()
    message(STATUS "Z3_DIR env var is not set and Z3 is not found.")
  endif()
endif()

# Use custom build of CTemplate if CT_DIR env var is specified
if(DEFINED ENV{CT_DIR})
  set(CT_DIR $ENV{CT_DIR})
  message(STATUS "CT_DIR env var is set: ${CT_DIR}.")
  include_directories(SYSTEM ${CT_DIR}/include)
  link_directories(SYSTEM ${CT_DIR}/lib)
else()
  message(STATUS "CT_DIR env var is not set.")
endif()

# Use custom build of Xerces-C++ XML Parser if XERCES_DIR env var is specified
if(DEFINED ENV{XERCES_DIR})
  message(STATUS "XERCES_DIR env var is set.")
  set(XERCES_DIR $ENV{XERCES_DIR})
  include_directories(SYSTEM ${XERCES_DIR}/include)
  link_directories(SYSTEM ${XERCES_DIR}/lib)
else()
  find_file(XERCES_C libxerces-c.so
    PATHS /usr/lib64 /usr/lib /usr/local/lib64
    REQUIRED
    DOC "Looking for libxerces-c.so installed to system paths")
  if (XERCES_C)
    message(STATUS "XERCES_C found: ${XERCES_C}")
  else()
    message(STATUS "XERCES_DIR env var is not set and XERCES_C is not found.")
  endif()
endif()

include_directories(SYSTEM lib/CLI11)
include_directories(SYSTEM lib/easyloggingpp)
include_directories(SYSTEM lib/json)
include_directories(SYSTEM lib/minisat)

link_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/lib/minisat)

add_subdirectory(lib)
add_subdirectory(src)

add_subdirectory(test)
