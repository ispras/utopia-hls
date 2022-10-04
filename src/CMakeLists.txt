# SPDX-License-Identifier: Apache-2.0

# The project executable
set(MAIN_TARGET umain)

# The project dependencies
set(MINISAT_LIBRARY minisat-lib-static)
set(STDCXXFS_LIBRARY stdc++fs)

# The project requires flex/bison.
find_package(FLEX REQUIRED)
find_package(BISON 3.0.0 REQUIRED)

# The configuration file w/ the project version.
configure_file(config.h.in config.h)

# The clang-tidy options.
set(CLANG_TIDY_EXCLUDE_LIST
  lexer.yy.cpp
  parser.tab.cpp
)

list(TRANSFORM CLANG_TIDY_EXCLUDE_LIST
  REPLACE .+ {"name":"\\0","lines":[[1,1]]}
)
list(JOIN CLANG_TIDY_EXCLUDE_LIST , CLANG_TIDY_LINE_FILTER)

set(CMAKE_CXX_CLANG_TIDY
  clang-tidy;
  -header-filter=src/.*;
  -line-filter=[${CLANG_TIDY_LINE_FILTER},{"name":".h"},{"name":".cpp"}];
  -checks=llvm-*,-llvm-header-guard;) # -warnings-as-errors;

# The compilation options.
add_compile_options(-g3 -Wall -Wignored-qualifiers -DYYERROR_VERBOSE)

add_subdirectory(util)
add_subdirectory(hls)
add_subdirectory(gate)
add_subdirectory(rtl)

# The project library.
add_library(${MAIN_LIBRARY})

# The libraries to link with.
target_link_libraries(${MAIN_LIBRARY}
  Utopia::Util
  Utopia::HLS
  Utopia::HILParser
  Utopia::Gate
  Utopia::RTL
  Utopia::RILParser
)

# The project executable.
add_executable(${MAIN_TARGET} main.cpp)

target_include_directories(${MAIN_TARGET}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${MAIN_TARGET}
  PRIVATE
    ${MAIN_LIBRARY}
    ${ELPP_LIBRARY}
    CLI
    Json
)
