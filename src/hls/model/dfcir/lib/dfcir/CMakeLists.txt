add_subdirectory(conversions)

add_mlir_dialect_library(MLIRDFCIR
        DFCIRDialect.cpp
        DFCIRTypes.cpp
        DFCIROpInterfaces.cpp
        DFCIROperations.cpp
        ${CONV_LIBS}

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/dfcir
        ${PROJECT_BINARY_DIR}/include/dfcir

        DEPENDS
        DFCIRPassesIncGen

        LINK_LIBS
        PUBLIC
        MLIRIR
        CIRCTFIRRTL
        MLIRParser
        MLIRRewrite
        MLIRTransformUtils

        PRIVATE
        LpSolve::LpSolve
        )


target_include_directories(MLIRDFCIR
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${CIRCT_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:include>
        )

add_library(DFCIR::MLIRDFCIR ALIAS MLIRDFCIR)

