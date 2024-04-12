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

        PRIVATE
        MLIRParser
        MLIRRewrite
        MLIRTransformUtils
        LpSolve::LpSolve
        )

target_include_directories(MLIRDFCIR
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
