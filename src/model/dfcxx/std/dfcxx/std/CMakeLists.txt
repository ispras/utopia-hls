## SPDX-License-Identifier: Apache-2.0

add_subdirectory(algebra)
add_subdirectory(crypto)

add_library(DFCXXSTD INTERFACE)

target_include_directories(DFCXXSTD
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/std>
    $<INSTALL_INTERFACE:${PROJECT_SOURCE_DIR}/std>
)

## MLIRDFCIR is ensured to be compiled beforehand.
target_link_libraries(DFCXXSTD
  INTERFACE Utopia::DFCXX
)

add_library(Utopia::DFCXXSTD ALIAS DFCXXSTD)
