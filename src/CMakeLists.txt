# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# The project name and version.
project(Utopia VERSION 0.0)

# The project executable and dependencies.
set(MAIN_TARGET umain)
set(Z3_LIBRARY z3)
set(XERCES_LIBRARY xerces-c)
set(LPSOLVE55_LIBRARY lpsolve55)
set(DL_LIBRARY dl)
set(COLAMD_LIBRARY colamd)
set(STDCXXFS_LIBRARY stdc++fs)
set(CTEMPLATE_LIBRARY ctemplate_nothreads)
set(MINISAT_LIBRARY minisat)

# The project requires flex/bison.
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Add mlir subproject
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hls/model/mlir)

# The configuration file w/ the project version.
configure_file(config.h.in config.h)

# The required C++ standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# The clang-tidy options.
set(FILTER_LEXER       {"name":"lexer.yy.cpp","lines":[[999999,999999]]})
set(FILTER_PARSER      {"name":"parser.tab.cpp","lines":[[999999,999999]]})
set(FILTER_PARSEUTILS  {"name":"ParseUtils.h","lines":[[999999,999999]]})
set(FILTER_SOLVER_CC   {"name":"Solver.cc","lines":[[999999,999999]]})
set(FILTER_SOLVER_H    {"name":"Solver.h","lines":[[999999,999999]]})
set(FILTER_SOLVERTYPES {"name":"SolverTypes.h","lines":[[999999,999999]]})
set(FILTER_ALG         {"name":"Alg.h","lines":[[999999,999999]]})
set(FILTER_ALLOC       {"name":"Alloc.h","lines":[[999999,999999]]})
set(FILTER_HEAP        {"name":"Heap.h","lines":[[999999,999999]]})
set(FILTER_INTTYPES    {"name":"IntTypes.h","lines":[[999999,999999]]})
set(FILTER_MAP         {"name":"Map.h","lines":[[999999,999999]]})
set(FILTER_VEC         {"name":"Vec.h","lines":[[999999,999999]]})
set(FILTER_XALLOC      {"name":"XAlloc.h","lines":[[999999,999999]]})
set(FILTER_OPTIONS     {"name":"Options.h","lines":[[999999,999999]]})

set(CMAKE_CXX_CLANG_TIDY
  clang-tidy;
  -header-filter=.;
  -line-filter=[${FILTER_LEXER},${FILTER_PARSER},${FILTER_PARSEUTILS},${FILTER_SOLVER_CC},${FILTER_SOLVER_H},${FILTER_SOLVERTYPES},${FILTER_ALG},${FILTER_ALLOC},${FILTER_HEAP},${FILTER_INTTYPES},${FILTER_MAP},${FILTER_VEC},${FILTER_XALLOC},${FILTER_OPTIONS},{"name":".h"},{"name":".cpp"}];
  -checks=llvm-*,-llvm-header-guard;) # -warnings-as-errors;

# The compilation options.
add_compile_options(-g3 -Wall -Wignored-qualifiers -DYYERROR_VERBOSE)

# The HLS lexer and parser.
add_custom_command(OUTPUT hls/parser/hil COMMAND ${CMAKE_COMMAND} -E make_directory hls/parser/hil)
add_custom_target(HlsHilParserDir DEPENDS hls/parser/hil)
bison_target(HlsHilParser hls/parser/hil/parser.y ${CMAKE_CURRENT_BINARY_DIR}/hls/parser/hil/parser.tab.cpp)
flex_target(HlsHilLexer hls/parser/hil/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/hls/parser/hil/lexer.yy.cpp)
add_flex_bison_dependency(HlsHilLexer HlsHilParser)

# The RTL lexer and parser.
add_custom_command(OUTPUT rtl/parser/ril COMMAND ${CMAKE_COMMAND} -E make_directory rtl/parser/ril)
add_custom_target(RtlRilParserDir DEPENDS rtl/parser/ril)
bison_target(RtlRilParser rtl/parser/ril/parser.y ${CMAKE_CURRENT_BINARY_DIR}/rtl/parser/ril/parser.tab.cpp)
flex_target(RtlRilLexer rtl/parser/ril/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/rtl/parser/ril/lexer.yy.cpp)
add_flex_bison_dependency(RtlRilLexer RtlRilParser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# The project library.
add_library(${MAIN_LIBRARY}

  # Manually-written sources.
  gate/debugger/encoder.cpp
  gate/debugger/checker.cpp
  gate/debugger/symexec.cpp
  gate/model/gate.cpp
  gate/model/gnet.cpp
  gate/model/gsymbol.cpp
  gate/model/signal.cpp
  hls/compiler/compiler.cpp
  hls/debugger/debugger.cpp
  hls/library/element_generator.cpp
  hls/library/element_core.cpp
  hls/library/internal/element_internal.cpp
  hls/library/ipxact_parser.cpp
  hls/library/library.cpp
  hls/mapper/config/hwconfig.cpp
  hls/mapper/mapper.cpp
  hls/model/model.cpp
  hls/model/transform.cpp
  hls/model/printer.cpp
  hls/parser/dfc/kernel.cpp
  hls/parser/dfc/stream.cpp
  hls/parser/dfc/internal/builder.cpp
  hls/parser/hil/builder.cpp
  hls/parser/hil/parser.cpp
  hls/scheduler/dijkstra.cpp
  hls/scheduler/flow_solver.cpp
  hls/scheduler/latency_solver.cpp
  hls/scheduler/lp_helper.cpp
  hls/scheduler/param_optimizer.cpp
  hls/scheduler/latency_balancer_base.h
  hls/scheduler/optimizers/abstract_optimizer.cpp
  hls/scheduler/optimizers/simple_optimizer.cpp
  hls/scheduler/optimizers/simulated_annealing_optimizer.cpp
  rtl/compiler/compiler.cpp
  rtl/library/flibrary.cpp
  rtl/model/event.cpp
  rtl/model/fsymbol.cpp
  rtl/model/net.cpp
  rtl/model/pnode.cpp
  rtl/model/vnode.cpp
  rtl/parser/ril/builder.cpp
  rtl/parser/ril/parser.cpp
  util/string.cpp

  # Automatically generated sources.
  "${FLEX_HlsHilLexer_OUTPUTS}"
  "${BISON_HlsHilParser_OUTPUTS}"
  "${FLEX_RtlRilLexer_OUTPUTS}"
  "${BISON_RtlRilParser_OUTPUTS}"
)

add_dependencies(${MAIN_LIBRARY} HlsHilParserDir RtlRilParserDir)

# The include directories.
target_include_directories(${MAIN_LIBRARY} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${MAIN_LIBRARY} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${MAIN_LIBRARY} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/hls/model/mlir/include)
target_include_directories(${MAIN_LIBRARY} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/hls/model/mlir/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# The libraries to link with.
target_link_libraries(${MAIN_LIBRARY}
  ${ELPP_LIBRARY}
  ${LPSOLVE55_LIBRARY}
  ${DL_LIBRARY}
  ${COLAMD_LIBRARY}
  ${Z3_LIBRARY}
  ${XERCES_LIBRARY}
  ${STDCXXFS_LIBRARY}
  ${CTEMPLATE_LIBRARY}
  ${MINISAT_LIBRARY}
  ${dialect_libs}
  ${translation_libs}
)

# The project executable.
add_executable(${MAIN_TARGET} main.cpp)
add_dependencies(${MAIN_TARGET} ${MAIN_LIBRARY})
target_link_libraries(${MAIN_TARGET} ${MAIN_LIBRARY})
