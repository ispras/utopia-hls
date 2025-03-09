## SPDX-License-Identifier: Apache-2.0

## Without disabling the "-Woverloaded-virtual" compiler flag
## warnings for internal CIRCT/MLIR code appear.
add_compile_options(-Wno-overloaded-virtual)

add_mlir_library(UtopiaDFCIRPasses
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRASAPSchedulerPass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRLinearSchedulerPass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRLPUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRPassesUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/DFCIRToFIRRTLPass.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FIRRTLStubsGeneratorPass.cpp

  DEPENDS
    DFCIRPassesIncGen

  LINK_LIBS
    PUBLIC
      CIRCTFIRRTL
      MLIRIR
      MLIRRewrite
      MLIRTransformUtils
      UtopiaDFCIR
    PRIVATE
      CTemplate::nothreads
      LpSolve::LpSolve
)

target_include_directories(UtopiaDFCIRPasses
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    ${CIRCT_INCLUDE_DIRS} ## TODO: Remove when CIRCT targets are fixed.
                          ## Issue #16 (https://github.com/ispras/utopia-hls/issues/16).
)

add_compile_definitions(
  TEMPLATES_PATH="${TEMPLATES_PATH}"
  STUBS_TEMPLATE_PATH="${TEMPLATES_PATH}/stubs.tpl"
)

add_library(Utopia::DFCIRPasses ALIAS UtopiaDFCIRPasses)
