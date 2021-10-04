# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)

# The project name and version.
project(Utopia VERSION 0.0)

# The project executable.
set(MAIN_TARGET umain)

# The project requires flex/bison.
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# The configuration file w/ the project version.
configure_file(config.h.in config.h)

# The required C++ standard.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

# The executable file.
add_executable(${MAIN_TARGET} main.cpp

  # Manually-written sources.
  gate/model/gate.cpp
  gate/model/gsymbol.cpp
  gate/model/netlist.cpp
  gate/model/signal.cpp
  hls/model/model.cpp
  hls/parser/hil/builder.cpp
  hls/parser/hil/parser.cpp
  rtl/compiler/compiler.cpp
  rtl/library/flibrary.cpp
  rtl/model/event.cpp
  rtl/model/fsymbol.cpp
  rtl/model/net.cpp
  rtl/model/vnode.cpp
  rtl/model/pnode.cpp
  rtl/parser/ril/builder.cpp
  rtl/parser/ril/parser.cpp
  util/utils.cpp

  # Automatically generated sources.
  "${FLEX_HlsHilLexer_OUTPUTS}"
  "${BISON_HlsHilParser_OUTPUTS}"
  "${FLEX_RtlRilLexer_OUTPUTS}"
  "${BISON_RtlRilParser_OUTPUTS}"
)

add_dependencies(${MAIN_TARGET} HlsHilParserDir RtlRilParserDir)

# The include directories.
target_include_directories(${MAIN_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${MAIN_TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# The libraries to link with.
target_link_libraries(${MAIN_TARGET} ${FLEX_LIBRARIES})
