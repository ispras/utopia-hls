## SPDX-License-Identifier: Apache-2.0

add_subdirectory(addconst)
add_subdirectory(idct)
add_subdirectory(matrixmul2)
add_subdirectory(movingaverage)
add_subdirectory(movingsum)
add_subdirectory(muxmul)
add_subdirectory(polynomial2)
add_subdirectory(scalar3)

set(EXAMPLES_TESTS
  ${ADDCONST_TESTS}
  ${IDCT_TESTS}
  ${MATRIXMUL2_TESTS}
  ${MOVINGAVERAGE_TESTS}
  ${MOVINGSUM_TESTS}
  ${MUXMUL_TESTS}
  ${POLYNOMIAL2_TESTS}
  ${SCALAR3_TESTS}
  PARENT_SCOPE
)
