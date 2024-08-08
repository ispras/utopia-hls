find_path(CTemplate_INCLUDE_DIR "ctemplate/template.h"
  PATH_SUFFIXES include)

if(NOT CTemplate_LIBRARY)
  find_library(CTemplate_LIBRARY ctemplate PATH_SUFFIXES lib)
endif()

if(NOT CTemplate_nothreads_LIBRARY)
  find_library(CTemplate_nothreads_LIBRARY ctemplate_nothreads
    PATH_SUFFIXES lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CTemplate
  REQUIRED_VARS CTemplate_LIBRARY CTemplate_INCLUDE_DIR)

if(CTemplate_FOUND)
  set(CTemplate_INCLUDE_DIRS ${CTemplate_INCLUDE_DIR})

  if(NOT CTemplate_LIBRARIES)
    set(CTemplate_LIBRARIES ${CTemplate_LIBRARY})
  endif()

  if(NOT TARGET CTemplate::CTemplate)
    add_library(CTemplate::CTemplate UNKNOWN IMPORTED)
    set_target_properties(CTemplate::CTemplate PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CTemplate_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "$<LINK_ONLY:Threads::Threads>"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${CTemplate_LIBRARY}"
    )
  endif()

  if((NOT TARGET CTemplate::nothreads) AND (CTemplate_nothreads_LIBRARY))
    add_library(CTemplate::nothreads UNKNOWN IMPORTED)
    set_target_properties(CTemplate::nothreads PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CTemplate_INCLUDE_DIRS}"
      IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
      IMPORTED_LOCATION "${CTemplate_nothreads_LIBRARY}"
    )
  endif()
endif()
