## SPDX-License-Identifier: Apache-2.0

add_subdirectory(dfcir)
add_subdirectory(dfcxx)

set(MODEL_TESTS
  ${DFCIR_TESTS}
  ${DFCXX_TESTS}
  PARENT_SCOPE
)
