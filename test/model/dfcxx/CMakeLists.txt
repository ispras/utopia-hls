## SPDX-License-Identifier: Apache-2.0

add_subdirectory(vars)

set(DFCXX_TESTS
  ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/output_formats.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/type_builder.cpp
  ${DFCXX_VARS_TESTS}
  ${CMAKE_CURRENT_SOURCE_DIR}/var_builder.cpp
  PARENT_SCOPE
)
