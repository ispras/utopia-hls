# SPDX-License-Identifier: Apache-2.0

# The project dependencies
set(MINISAT_LIBRARY minisat-lib-static)
set(STDCXXFS_LIBRARY stdc++fs)

# The configuration file w/ the project version.
configure_file(config.h.in config.h)

# The clang-tidy options.
##set(CLANG_TIDY_EXCLUDE_LIST
##  lexer.yy.cpp
##  parser.tab.cpp
##)

##list(TRANSFORM CLANG_TIDY_EXCLUDE_LIST
##  REPLACE .+ {"name":"\\0","lines":[[1,1]]}
##)
##list(JOIN CLANG_TIDY_EXCLUDE_LIST , CLANG_TIDY_LINE_FILTER)

##set(CMAKE_CXX_CLANG_TIDY
##  clang-tidy;
##  -header-filter=src/.*;
##  -line-filter=[${CLANG_TIDY_LINE_FILTER},{"name":".h"},{"name":".cpp"}];
##  -checks=llvm-*,-llvm-header-guard;) # -warnings-as-errors;

# The compilation options.
add_compile_options(-g3 -Wall -Wignored-qualifiers -DYYERROR_VERBOSE)

add_subdirectory(utils)
add_subdirectory(hls)

# The project library.
##add_library(ueda)

# The libraries to link with.
##target_link_libraries(ueda
  #Utopia::Utils
  #Utopia::HLS
#)

##===----------------------------------------------------------------------===//
## User-defined code compilation

add_library(uuser ${SRC_FILES})
target_include_directories(uuser
  PUBLIC
    ${INCLUDE_DIRS})

target_link_libraries(uuser
  PUBLIC
    Utopia::DFCXX)
add_library(Utopia::User ALIAS uuser)

##===----------------------------------------------------------------------===//

add_executable(umain main.cpp)

target_compile_definitions(umain
  PUBLIC
    JSON_CONFIG_PATH="${CMAKE_SOURCE_DIR}/config.json"
)

set_target_properties(umain PROPERTIES OUTPUT_NAME "${OUT}")

target_include_directories(umain
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(umain
  PRIVATE
    Utopia::User
    Utopia::DFCXX
    easyloggingpp
    CLI
    Json
)
