//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include "gtest/gtest.h"

#include "hls/library/library.h"
#include "hls/library/ipxact_parser.h"

using namespace eda::hls::library;

int parserCatalogTest(const std::string &libraryPath,
                      const std::string &catalogPath) {
  const auto paths = IPXACTParser::get().parseCatalog(libraryPath, catalogPath);
  for (const auto &path : paths) {
    std::cout << "Path to component: " << path << std::endl;
  }
  return 0;
}

int parserComponentTest(const std::string &libraryPath,
                        const std::string &componentPath) {
  auto metaElement = IPXACTParser::get().parseComponent(libraryPath,
                                                        componentPath);
  std::cout << "MetaElement: " << metaElement->name << " " << std::endl;
  std::cout << "Library: " << metaElement->libraryName << std::endl;
  std::cout << "Ports:" << std::endl;
  for (const auto &port : metaElement->ports) {
    std::cout << (port.direction ? "out " : "in ") << port.name << " ";
    std::cout << (port.width > 1 ? ("[" + std::to_string(port.width - 1) + 
                                    ":" + "0" + "]") : "");
    if (port.width < 1) {
      std::cout << "Parameter width " << port.param.getName() << "; ";
      std::cout << "Constraints: " << "[" << port.param.getMin() << ",";
      std::cout << " " << port.param.getMax() << "];";
    }
    std::cout << std::endl;
  }
  std::cout << "Parameters:" << std::endl;
  for (const auto &pair : metaElement->params.getAll()) {
    std::cout << "Name: " << pair.second.getName() << std::endl;
    std::cout << "Constraints: " << "[" << pair.second.getMin() << ",";
    std::cout << " " << pair.second.getMax() << "]" << std::endl;
    std::cout << "Value: " << pair.second.getValue() << std::endl;
  }
  return 0;
}

/*TEST(LibraryTest, ParserComponentEmptyDocumentTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "empty_document.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentEmptyXmlDocumentTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "empty_xml_document.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentMissingComponentTagTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_component_tag.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentMissingClosingTagComponentTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_closing_tag_component.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentMissingSlashInClosingTagComponentTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_slash_in_closing_tag_component.xml"),
                                 0);
}*/

/*TEST(LibraryTest, ParserComponentEmptyComponentTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/", 
                                "empty_component.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentClosingTagAtTheBeginningTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "closing_tag_at_the_beginning.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentMissingValueInTagNotUsedTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_value_in_tag_not_used.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentMissingValueInTagUsedTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_value_in_tag_used.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentDoubleTagNameTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "double_tag_name.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentMissingClosingBracketTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_closing_bracket.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentIncorrectValueInTagDirectionTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "incorrect_value_in_tag_direction.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentParameterPortWidthTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "parameter_port_width.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentNegativeValueInTagLeftTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "negative_value_in_tag_left.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentIncorrectMarginsTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "incorrect_margins.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentNoVendorExtensionsTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "no_vendor_extensions.xml"), 0);
}

TEST(LibraryTest, ParserComponentEmptyTagPortsTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "empty_tag_ports.xml"), 0);
}

TEST(LibraryTest, ParserComponentMissingTagPortsTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_tag_ports.xml"), 0);
}

TEST(LibraryTest, ParserComponentEmptyTagVectorsTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "empty_tag_vectors.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentMissingValueInTagNameTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_value_in_tag_name.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentMissingTagVectorsTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_tag_vectors.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentMissingTagDirectionTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_tag_direction.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentEmptyTagParametersTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "empty_tag_parameters.xml"), 0);
}

TEST(LibraryTest, ParserComponentIncorrectPathToSchemaTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "incorrect_path_to_schema.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentMissingTagValueTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "missing_tag_value.xml"), 0);
}*/

TEST(LibraryTest, ParserComponentPortTagInPortTagTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "port_tag_in_port_tag.xml"), 0);
}

/*TEST(LibraryTest, ParserComponentNewlineInTagTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "newline_in_tag.xml"), 0);
}*/

/*TEST(LibraryTest, ParserComponentIncorrectValueInTagValueTest) {
  EXPECT_EQ(parserComponentTest("test/data/ipx/test/component/",
                                "incorrect_value_in_tag_value.xml"), 0);
}*/



/*TEST(LibraryTest, ParserCatalogEmptyDocumentTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "empty_document.xml"), 0);
}*/

TEST(LibraryTest, ParserCatalogEmptyCatalogTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "empty_catalog.xml"), 0);
}

/*TEST(LibraryTest, ParserCatalogEmptyXmlDocumentTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "empty_xml_document.xml"), 0);
}*/

TEST(LibraryTest, ParserCatalogDoubleTagNameTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "double_tag_name.xml"), 0);
}

TEST(LibraryTest, ParserCatalogMissingTagComponentsTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_tag_components.xml"), 0);
}

TEST(LibraryTest, ParserCatalogIpxactFileTagInIpxactFileTagTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "ipxactFile_tag_in_ipxactFile_tag.xml"), 0);
}

/*TEST(LibraryTest, ParserCatalogMissingClosingTagVendorTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_closing_tag_vendor.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingValueTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_value.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingSlashInClosingTagTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_slash_in_closing_tag.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogClosingTagAtTheBeginningTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "closing_tag_at_the_beginning.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingTagCatalogTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_tag_catalog.xml"), 0);
}*/

TEST(LibraryTest, ParserCatalogMissingComponentsTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_components.xml"), 0);
}

/*TEST(LibraryTest, ParserCatalogMissingClosingBracketTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_closing_bracket.xml"), 0);
}*/

TEST(LibraryTest, ParserCatalogMissingTagIpxactFileTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_tag_ipxactFile.xml"), 0);
}

/*TEST(LibraryTest, ParserCatalogMissingClosingTagNoPairTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_closing_tag_no_pair.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingAttributeValueEqualTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_attribute_value_equal.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingAttributeValueNoEqualTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_attribute_value_no_equal.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogEmptyIpxactFileTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "empty_ipxactFile.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingQuotesInAttributeValueTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_quotes_in_attribute_value.xml"), 0);
}*/

/*TEST(LibraryTest, ParserCatalogMissingClosingQuotesInAttributeValueTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "missing_closing_quotes_in_attribute_value.xml"),
                               0);
}*/

/*TEST(LibraryTest, ParserCatalogNewLineInTagTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "newline_in_tag.xml"), 0);
}*/

TEST(LibraryTest, ParserCatalogNewLineInTagValueTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "newline_in_tag_value.xml"), 0);
}

TEST(LibraryTest, ParserCatalogQuotesInTagValueNotClosedTest) {
  EXPECT_EQ(parserCatalogTest("test/data/ipx/test/catalog/",
                              "quotes_in_tag_value_not_closed.xml"), 0);
}
