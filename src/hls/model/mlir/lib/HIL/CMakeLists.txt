add_mlir_dialect_library(MLIRHIL
        API.cpp
        Dialect.cpp
        Ops.cpp
        Combine.cpp
        Model.cpp
        Dumper.cpp
        Utils.cpp
        ${PROJECT_SOURCE_DIR}/../../parser/hil/builder.cpp
        ${PROJECT_SOURCE_DIR}/../model.cpp
        ${PROJECT_SOURCE_DIR}/../transform.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/HIL

        DEPENDS
        MLIROpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
