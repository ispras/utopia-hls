project(hil-dialect LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# FIXME resolve circular dependencies w/ model and remove the direct inclusion
include_directories(${Utopia_SOURCE_DIR}/src)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(hil-opt)
add_subdirectory(hil-translate)
