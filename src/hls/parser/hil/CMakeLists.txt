bison_target(HlsHilParser parser.y ${CMAKE_CURRENT_BINARY_DIR}/parser.tab.cpp)
flex_target(HlsHilLexer lexer.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.yy.cpp)
add_flex_bison_dependency(HlsHilLexer HlsHilParser)

add_library(HILParser OBJECT
  builder.cpp
  parser.cpp
  ${FLEX_HlsHilLexer_OUTPUTS}
  ${BISON_HlsHilParser_OUTPUTS}
)
add_library(Utopia::HILParser ALIAS HILParser)

target_include_directories(HILParser
  PUBLIC
    ${PROJECT_SOURCE_DIR}/src

  PRIVATE
    ${PROJECT_BINARY_DIR}/src
)

target_link_libraries(HILParser
  PUBLIC
    MLIRHIL
)
