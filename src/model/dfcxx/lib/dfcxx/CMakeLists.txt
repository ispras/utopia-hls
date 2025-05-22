## SPDX-License-Identifier: Apache-2.0

add_subdirectory(types)
add_subdirectory(vars)

## By default LLVM, MLIR and DFCIR are compiled without RTTI, but not DFCxx.
## So using core Pass functionality from MLIR in converter.cpp causes
## linker-level errors ("undefined reference to `typeinfo for mlir::Pass'").
## The following line explicitly disables RTTI for DFCxx.
add_definitions("-fno-rtti")

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/channel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/control.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dfcir_builder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dfcir_processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/io.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_meta.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel_storage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/offset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/simulator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type_builder.cpp
  ${TYPES_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/var_builder.cpp
  ${VARS_SOURCES}
)

add_library(DFCXX STATIC
  ${SOURCES}
)

target_include_directories(DFCXX
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

add_compile_definitions(
  TEMPLATES_PATH="${TEMPLATES_PATH}"
  DOT_TEMPLATE_PATH="${TEMPLATES_PATH}/dot.tpl"
  VCD_TEMPLATE_PATH="${TEMPLATES_PATH}/vcd.tpl"
)

## MLIRDFCIR is ensured to be compiled beforehand.
target_link_libraries(DFCXX
  PRIVATE $<LINK_ONLY:MLIRPass>
  PRIVATE $<LINK_ONLY:MLIRTransforms>
  PRIVATE $<LINK_ONLY:Utopia::DFCIR>
  PRIVATE $<LINK_ONLY:Utopia::DFCIRPasses>
  PRIVATE $<LINK_ONLY:CIRCTHW>
  PRIVATE $<LINK_ONLY:CIRCTHWTransforms>
  PRIVATE $<LINK_ONLY:CIRCTSV>
  PRIVATE $<LINK_ONLY:CIRCTSVTransforms>
  PRIVATE $<LINK_ONLY:CIRCTFIRRTLTransforms>
  PRIVATE $<LINK_ONLY:CIRCTFIRRTLToHW>
  PRIVATE $<LINK_ONLY:CIRCTSeqToSV>
  PRIVATE $<LINK_ONLY:CIRCTExportVerilog>
  PRIVATE $<LINK_ONLY:CTemplate::nothreads>
  PRIVATE $<LINK_ONLY:LLVMSupport>
)

add_library(Utopia::DFCXX ALIAS DFCXX)
