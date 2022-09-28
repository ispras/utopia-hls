# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(Utopia VERSION 0.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

#find_package(LpSolve REQUIRED)
include(cmake/MLIR.cmake)

set(MAIN_LIBRARY ueda)          # Used by {src,test}/CMakeLists.txt
set(ELPP_LIBRARY easyloggingpp) # Used by {lib,src}/CMakeLists.txt
#set(LORINA_LIBRARY lorina)

set(SYSLIB_PATHS /usr/lib64 /usr/lib
                 /usr/local/lib /usr/local/lib64
                 /usr/lib/x86_64-linux-gnu
                 /opt/homebrew/Cellar/xerces-c/3.2.3/lib #!!!
                 /opt/homebrew/Cellar/lp_solve/5.5.2.11/lib) #!!!

# Check if UTOPIA_HOME env var is set
if(DEFINED ENV{UTOPIA_HOME})
  message(STATUS "UTOPIA_HOME: $ENV{UTOPIA_HOME}")
else()
  message(FATAL_ERROR "UTOPIA_HOME env var is not set!")
endif()

# Looking for CTemplate library
if(DEFINED ENV{CT_DIR})
  message(STATUS "CT_DIR: $ENV{CT_DIR}")
  set(CT_DIR $ENV{CT_DIR})
  include_directories(SYSTEM ${CT_DIR}/include)
  link_directories(SYSTEM ${CT_DIR}/lib)
else()
  find_file(CTEMPLATE libctemplate.dylib
    PATHS ${SYSLIB_PATHS}
    REQUIRED
    DOC "Looking for libctemplate.dylib installed to system paths")

    message(FATAL_ERROR "CT_DIR env var is not set and CTEMPLATE is not found.")
endif()

# Looking for Xerces library
find_file(XERCES_C libxerces-c.dylib
  PATHS ${SYSLIB_PATHS}
  REQUIRED
  DOC "Looking for libxerces-c.dylib installed to system paths")
  #!!!!
  include_directories(SYSTEM /opt/homebrew/Cellar/xerces-c/3.2.3/include)
if(XERCES_C)
  message(STATUS "XERCES_C found: ${XERCES_C}")
else()
  message(FATAL_ERROR "XERCES_C lib is not found.")
endif()

# Looking for Z3 library
if(DEFINED ENV{Z3_DIR})
  message(STATUS "Z3_DIR: $ENV{Z3_DIR}")
  set(Z3_DIR $ENV{Z3_DIR})
  include_directories(SYSTEM ${Z3_DIR}/include)
  link_directories(SYSTEM ${Z3_DIR}/lib)
else()
  find_file(Z3 libz3.so
    PATHS ${SYSLIB_PATHS}
    REQUIRED
    DOC "Looking for libz3.so installed to system paths")
  if(Z3)
    message(STATUS "Z3 found: ${Z3}")
  else()
    message(FATAL_ERROR "Z3_DIR env var is not set and Z3 is not found.")
  endif()
endif()

#set(CMAKE_C_COMPILER "/opt/homebrew/opt/gcc-12")
#set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/g++-12")

include_directories(SYSTEM lib/CLI11)
include_directories(SYSTEM lib/easyloggingpp)
include_directories(SYSTEM lib/json)
include_directories(SYSTEM lib/minisat)
include_directories(SYSTEM lib/lorina/include)
include_directories(SYSTEM lib/lorina/lib/fmt)

link_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/lib/minisat)

add_subdirectory(lib)
add_subdirectory(src)

add_subdirectory(test)
