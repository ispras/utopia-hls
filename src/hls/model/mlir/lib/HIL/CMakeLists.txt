add_mlir_dialect_library(MLIRHIL
        HILDialect.cpp
        HILOps.cpp
        HILCombine.cpp
        HILModel.cpp
        HILDumper.cpp
        ${PROJECT_SOURCE_DIR}/../../parser/hil/builder.cpp
        ${PROJECT_SOURCE_DIR}/../model.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/HIL

        DEPENDS
        MLIRHILOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
