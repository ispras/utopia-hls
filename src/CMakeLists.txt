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
