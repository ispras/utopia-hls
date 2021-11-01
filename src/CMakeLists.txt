# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# The project name and version.
project(Utopia VERSION 0.0)

# The project library and executable.
set(MAIN_LIBRARY ueda)
set(MAIN_TARGET umain)
set(Z3_LIBRARY z3)
set(LPSOLVE55_LIBRARY lpsolve55)
set(DL_LIBRARY dl)
set(COLAMD_LIBRARY colamd)

# Use custom build of Z3 if Z3_DIR env var is specified
if(DEFINED ENV{Z3_DIR})
  message(STATUS "Z3_DIR env var is set.")
  set(Z3_DIR $ENV{Z3_DIR})
  set(ENV{LD_LIBRARY_PATH} "$ENV{LD_LIBRARY_PATH}:$Z3_DIR")
  set(CMAKE_CXX_FLAGS "-I${Z3_DIR}/include")
  set(CMAKE_EXE_LINKER_FLAGS "-L${Z3_DIR}/lib")
else()
  message(STATUS "Z3_DIR env var is not set.")
endif()

# The project requires flex/bison.
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# The configuration file w/ the project version.
configure_file(config.h.in config.h)

# The required C++ standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# The clang-tidy options.
set(FILTER_LEXER  {"name":"lexer.yy.cpp","lines":[[999999,999999]]})
set(FILTER_PARSER {"name":"parser.tab.cpp","lines":[[999999,999999]]})

set(CMAKE_CXX_CLANG_TIDY
  clang-tidy;
  -header-filter=.;
  -line-filter=[${FILTER_LEXER},${FILTER_PARSER},{"name":".h"},{"name":".cpp"}];
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
  gate/model/gate.cpp
  gate/model/gsymbol.cpp
  gate/model/netlist.cpp
  gate/model/signal.cpp
  hls/debugger/debugger.cpp
  hls/library/library.cpp
  hls/compiler/compiler.cpp
  hls/model/model.cpp
  hls/model/printer.cpp
  hls/parser/hil/builder.cpp
  hls/parser/hil/parser.cpp
  hls/scheduler/dijkstra.cpp
  hls/scheduler/lp_helper.cpp
  hls/scheduler/scheduler.cpp
  hls/scheduler/solver.cpp
  rtl/compiler/compiler.cpp
  rtl/library/flibrary.cpp
  rtl/model/event.cpp
  rtl/model/fsymbol.cpp
  rtl/model/net.cpp
  rtl/model/vnode.cpp
  rtl/model/pnode.cpp
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

# The libraries to link with.
target_link_libraries(${MAIN_LIBRARY} ${LPSOLVE55_LIBRARY} ${DL_LIBRARY} ${COLAMD_LIBRARY} ${Z3_LIBRARY})

# The project executable.
add_executable(${MAIN_TARGET} main.cpp)
add_dependencies(${MAIN_TARGET} ${MAIN_LIBRARY})
target_link_libraries(${MAIN_TARGET} ${MAIN_LIBRARY})
