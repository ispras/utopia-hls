find_path(Z3_INCLUDE_DIR z3.h PATH_SUFFIXES include)

# Allow Z3_LIBRARY to be set manually
if(NOT Z3_LIBRARY)
    find_library(Z3_LIBRARY z3 PATH_SUFFIXES lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Z3 REQUIRED_VARS Z3_LIBRARY Z3_INCLUDE_DIR)

if(Z3_FOUND)
    set(Z3_INCLUDE_DIRS ${Z3_INCLUDE_DIR})

    if(NOT Z3_LIBRARIES)
        set(Z3_LIBRARIES ${Z3_LIBRARY})
    endif()

    if(NOT TARGET Z3::Z3)
        add_library(Z3::Z3 UNKNOWN IMPORTED)
        set_target_properties(Z3::Z3 PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Z3_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${Z3_LIBRARY}"
        )
    endif()
endif()
