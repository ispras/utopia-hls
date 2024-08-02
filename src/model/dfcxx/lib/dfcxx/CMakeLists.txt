add_subdirectory(IRbuilders)
add_subdirectory(types)
add_subdirectory(typebuilders)
add_subdirectory(vars)
add_subdirectory(varbuilders)

## By default LLVM, MLIR and DFCIR are compiled without RTTI, but not DFCxx.
## So using core Pass functionality from MLIR in converter.cpp causes
## linker-level errors ("undefined reference to `typeinfo for mlir::Pass'").
## The following lines explicitly disable RTTI for DFCxx.
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
  set(DISABLE_RTTI_FLAG "/GR-")
else()
  set(DISABLE_RTTI_FLAG "-fno-rtti")
endif()
add_definitions(${DISABLE_RTTI_FLAG})

set(SOURCES
  kernel.cpp
  ${IR_BUILDERS_SOURCES}
  ${TYPES_SOURCES}
  ${VARS_SOURCES}
  ${VAR_BUILDERS_SOURCES}
  ${TYPE_BUILDERS_SOURCES}
  converter.cpp
  kernstorage.cpp
  io.cpp
  offset.cpp
  constant.cpp
  control.cpp
  graph.cpp
  node.cpp
  channel.cpp
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

## MLIRDFCIR is ensured to be compiled beforehand.
target_link_libraries(DFCXX
  PRIVATE $<LINK_ONLY:MLIRPass>
  PRIVATE $<LINK_ONLY:Utopia::MLIRDFCIR>
  PRIVATE $<LINK_ONLY:CIRCTHW>
  PRIVATE $<LINK_ONLY:CIRCTSV>
  PRIVATE $<LINK_ONLY:CIRCTFIRRTLToHW>
  PRIVATE $<LINK_ONLY:CIRCTSeqToSV>
  PRIVATE $<LINK_ONLY:CIRCTExportVerilog>
)

add_library(Utopia::DFCXX ALIAS DFCXX)
