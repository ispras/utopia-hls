add_subdirectory(conversions)

add_mlir_dialect_library(MLIRDFCIR
        DFCIRDialect.cpp
        DFCIRTypes.cpp
        DFCIROpInterfaces.cpp
        DFCIROperations.cpp
        ${CONV_LIBS}

        DEPENDS
        DFCIRPassesIncGen
        )

target_include_directories(MLIRDFCIR
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
        ${CIRCT_INCLUDE_DIRS} ## TODO: Remove when CIRCT targets are fixed.
        )

target_link_libraries(MLIRDFCIR
        PUBLIC
        MLIRIR
        CIRCTFIRRTL
        MLIRParser
        MLIRRewrite
        MLIRTransformUtils
        
        PRIVATE
        LpSolve::LpSolve
        )

add_library(Utopia::MLIRDFCIR ALIAS MLIRDFCIR)
