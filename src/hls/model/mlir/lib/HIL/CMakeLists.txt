add_mlir_dialect_library(MLIRHIL
        API.cpp
        Dialect.cpp
        Ops.cpp
        Combine.cpp
        Model.cpp
        Dumper.cpp
        Utils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/HIL

        DEPENDS
        MLIROpsIncGen

        LINK_LIBS
          PUBLIC
            MLIRIR

          PRIVATE
            MLIRArithmetic
            MLIRParser
            MLIRRewrite
            MLIRTransformUtils
        )

target_include_directories(MLIRHIL
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
