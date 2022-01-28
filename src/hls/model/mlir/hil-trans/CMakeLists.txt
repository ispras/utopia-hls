get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRHIL
        )
add_llvm_executable(hil-trans hil-trans.cpp)

llvm_update_compile_flags(hil-trans)
target_link_libraries(hil-trans PRIVATE ${LIBS})

mlir_check_all_link_libraries(hil-trans)
