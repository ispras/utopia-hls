cmake_minimum_required(VERSION 3.10)

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
add_compile_options(-g3 -Wall -DYYERROR_VERBOSE)

# The HLS lexer and parser.
add_custom_command(OUTPUT hls/parser COMMAND ${CMAKE_COMMAND} -E make_directory hls/parser)
add_custom_target(HlsParserDir DEPENDS hls/parser)
bison_target(HlsParser hls/parser/parser.y ${CMAKE_CURRENT_BINARY_DIR}/hls/parser/parser.tab.cpp)
flex_target(HlsLexer hls/parser/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/hls/parser/lexer.yy.cpp)
add_flex_bison_dependency(HlsLexer HlsParser)

# The RTL lexer and parser.
add_custom_command(OUTPUT rtl/parser COMMAND ${CMAKE_COMMAND} -E make_directory rtl/parser)
add_custom_target(RtlParserDir DEPENDS rtl/parser)
bison_target(RtlParser rtl/parser/parser.y ${CMAKE_CURRENT_BINARY_DIR}/rtl/parser/parser.tab.cpp)
flex_target(RtlLexer rtl/parser/lexer.l ${CMAKE_CURRENT_BINARY_DIR}/rtl/parser/lexer.yy.cpp)
add_flex_bison_dependency(RtlLexer RtlParser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

# The executable file.
add_executable(${MAIN_TARGET} main.cpp

  # Manually-written sources.
  gate/model/gate.cpp
  gate/model/gsymbol.cpp
  gate/model/netlist.cpp
  gate/model/signal.cpp
  hls/model/model.cpp
  hls/parser/builder.cpp
  hls/parser/parser.cpp
  rtl/compiler/compiler.cpp
  rtl/library/flibrary.cpp
  rtl/model/event.cpp
  rtl/model/fsymbol.cpp
  rtl/model/net.cpp
  rtl/model/vnode.cpp
  rtl/model/pnode.cpp
  rtl/parser/builder.cpp
  rtl/parser/parser.cpp
  util/utils.cpp

  # Automatically generated sources.
  "${FLEX_HlsLexer_OUTPUTS}"
  "${BISON_HlsParser_OUTPUTS}"
  "${FLEX_RtlLexer_OUTPUTS}"
  "${BISON_RtlParser_OUTPUTS}"
)

add_dependencies(${MAIN_TARGET} HlsParserDir RtlParserDir)

# The include directories.
target_include_directories(${MAIN_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${MAIN_TARGET} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# The libraries to link with.
target_link_libraries(${MAIN_TARGET} ${FLEX_LIBRARIES})
