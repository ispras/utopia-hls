get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRHIL
        )
add_llvm_executable(hil-test test_transforms.cpp)

llvm_update_compile_flags(hil-test)

mlir_check_all_link_libraries(hil-test)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (ADD_GTEST_DIR OR PROJECT_IS_TOP_LEVEL)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/googletest)
endif()

enable_testing()
target_include_directories(
  hil-test
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/HIL
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(hil-test PRIVATE ${LIBS} gtest_main)
target_link_directories(hil-test
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../lib/HIL
  ${LLVM_BUILD_LIBRARY_DIR}
)

include(GoogleTest)
gtest_discover_tests(hil-test)
