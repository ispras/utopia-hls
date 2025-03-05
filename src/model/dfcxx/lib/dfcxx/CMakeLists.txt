add_subdirectory(types)
add_subdirectory(vars)

## By default LLVM, MLIR and DFCIR are compiled without RTTI, but not DFCxx.
## So using core Pass functionality from MLIR in converter.cpp causes
## linker-level errors ("undefined reference to `typeinfo for mlir::Pass'").
## The following line explicitly disables RTTI for DFCxx.
add_definitions("-fno-rtti")

set(SOURCES
  channel.cpp
  constant.cpp
  control.cpp
  converter.cpp
  graph.cpp
  io.cpp
  irbuilder.cpp
  kernel.cpp
  kernmeta.cpp
  kernstorage.cpp
  node.cpp
  offset.cpp
  simulator.cpp
  typebuilder.cpp
  ${TYPES_SOURCES}
  utils.cpp
  varbuilder.cpp
  ${VARS_SOURCES}
)

add_library(DFCXX STATIC
  ${SOURCES}
)

target_include_directories(DFCXX
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/includeDev>
)

set(TEMPLATES_PATH "${PROJECT_SOURCE_DIR}/templates")

add_compile_definitions(
  TEMPLATES_PATH="${TEMPLATES_PATH}"
  DOT_TEMPLATE_PATH="${TEMPLATES_PATH}/dot.tpl"
  VCD_TEMPLATE_PATH="${TEMPLATES_PATH}/vcd.tpl"
)

## MLIRDFCIR is ensured to be compiled beforehand.
target_link_libraries(DFCXX
  PRIVATE $<LINK_ONLY:MLIRPass>
  PRIVATE $<LINK_ONLY:Utopia::MLIRDFCIR>
  PRIVATE $<LINK_ONLY:CIRCTHW>
  PRIVATE $<LINK_ONLY:CIRCTSV>
  PRIVATE $<LINK_ONLY:CIRCTFIRRTLToHW>
  PRIVATE $<LINK_ONLY:CIRCTSeqToSV>
  PRIVATE $<LINK_ONLY:CIRCTExportVerilog>
  PRIVATE $<LINK_ONLY:CTemplate::nothreads>
  PRIVATE $<LINK_ONLY:LLVMSupport>
)

add_library(Utopia::DFCXX ALIAS DFCXX)
