//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/element_core.h"
#include "hls/library/element_generator.h"
#include "hls/library/element_internal.h"
#include "hls/library/ipxact_parser.h"
#include "hls/library/library.h"
#include "util/singleton.h"

#include <cassert>
#include <ctemplate/template.h>
#include <iostream>
#include <vector>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/DOMLSParserImpl.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>


using namespace xercesc;
using namespace std;

namespace eda::hls::library {

class IPXACTParser final : public Singleton<IPXACTParser> {
  friend Singleton<IPXACTParser>;

public:
  void initialize();
  void finalize();

  bool hasComponent(const std::string &name);

  void parseCatalog(const std::string &libPath, const std::string &catalogPath);
  std::shared_ptr<MetaElement> parseComponent(const std::string &name);

private:
  IPXACTParser() {}

  std::map<std::string, std::string> comp_fnames;

  std::string libraryPath;

};

} // namespace eda::hls::library
