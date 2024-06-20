# SPDX-License-Identifier: Apache-2.0

# The configuration file w/ the project version.
configure_file(config.h.in config.h)

# The compilation options.
add_compile_options(-g3 -Wall -Wignored-qualifiers)

add_subdirectory(model)

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
