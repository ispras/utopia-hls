//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "util/singleton.h"

#include <ctemplate/template.h>
#include <iostream>
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

  void parseCatalog(const std::string &libPath, const std::string &catalogPath);
  std::shared_ptr<MetaElement> parseComponent(const std::string &name);

private:
  IPXACTParser() {}

  std::map<std::string, std::string> comp_fnames;

  std::string libPath;

};

} // namespace eda::hls::library
