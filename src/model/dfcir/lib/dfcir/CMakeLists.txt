## SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(UtopiaDFCIR
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRDialect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRTypes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIROpInterfaces.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIROperations.cpp

  DEPENDS
    DFCIRPassesIncGen

  LINK_LIBS
    PUBLIC
      MLIRIR
)

target_include_directories(UtopiaDFCIR
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)

add_library(Utopia::DFCIR ALIAS UtopiaDFCIR)

add_subdirectory(passes)
