add_library(Utils OBJECT
  fm.cpp
  partition_hgraph.cpp
  string.cpp
)
target_include_directories(Utils PUBLIC ${PROJECT_SOURCE_DIR}/src)

add_library(Utopia::Utils ALIAS Utils)