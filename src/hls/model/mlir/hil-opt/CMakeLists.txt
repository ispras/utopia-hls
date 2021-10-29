get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRHIL
        )
add_llvm_executable(hil-opt hil-opt.cpp)

llvm_update_compile_flags(hil-opt)
target_link_libraries(hil-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(hil-opt)
