# SPDX-License-Identifier: Apache-2.0

add_library(CLI INTERFACE)
target_include_directories(CLI INTERFACE CLI11)

add_library(Json INTERFACE)
target_include_directories(Json INTERFACE json)
target_compile_options(Json
  INTERFACE
    -Wno-covered-switch-default
)

add_subdirectory(easyloggingpp)
