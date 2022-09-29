# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(Utopia VERSION 0.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(cmake/MLIR.cmake)

set(MAIN_LIBRARY ueda)          # Used by {src,test}/CMakeLists.txt
set(ELPP_LIBRARY easyloggingpp) # Used by {lib,src}/CMakeLists.txt

# Check if UTOPIA_HOME env var is set
if(DEFINED ENV{UTOPIA_HOME})
  message(STATUS "UTOPIA_HOME: $ENV{UTOPIA_HOME}")
else()
  message(FATAL_ERROR "UTOPIA_HOME env var is not set!")
endif()

include(UtopiaFindHelper)
set_local_from_env(CTemplate_ROOT CT_DIR)
set_local_from_env(XercesC_ROOT XERCES_DIR)
set_local_from_env(Z3_ROOT Z3_DIR)

find_package(LpSolve REQUIRED)
find_package(Z3 REQUIRED)
find_package(XercesC REQUIRED)
find_package(CTemplate REQUIRED COMPONENTS nothreads)

include_directories(SYSTEM lib/CLI11)
include_directories(SYSTEM lib/easyloggingpp)
include_directories(SYSTEM lib/json)

add_subdirectory(lib)
add_subdirectory(src)

add_subdirectory(test)
