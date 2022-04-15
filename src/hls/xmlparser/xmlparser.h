//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2021 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/DOMLSParserImpl.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <ctemplate/template.h>

#include <iostream>

using namespace xercesc;
using namespace std;

namespace eda::hls::xmlparser {
  struct XMLParser final {

  const char* catalog_fname;

  std::vector<const char*> comp_fnames;

  void parseIPXACT();
  void parseCatalog();
  void parseComponent(int index);

  XMLParser(const char* catalog_fname);
  ~XMLParser();
};
} // namespace eda::hls::xmlparser
