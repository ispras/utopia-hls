//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/ipxact_parser.h"
#include "hls/library/library.h"
#include "util/singleton.h"

#include <xercesc/util/XMLString.hpp>

using namespace xercesc;

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
  IPXACTParser() {
    XMLPlatformUtils::Initialize();
    nameAttr       = XMLString::transcode("name");
    ipxPortTag     = XMLString::transcode("ipxact:port");
    ipxNameTag     = XMLString::transcode("ipxact:name");
    ipxVlnvTag     = XMLString::transcode("ipxact:vlnv");
    ipxDirectTag   = XMLString::transcode("ipxact:direction");
    ipxVectTag     = XMLString::transcode("ipxact:vector");
    ipxLeftTag     = XMLString::transcode("ipxact:left");
    ipxRightTag    = XMLString::transcode("ipxact:right");
    k2ParamTag     = XMLString::transcode("kactus2:parameter");
    k2NameTag      = XMLString::transcode("kactus2:name");
    k2ValueTag     = XMLString::transcode("kactus2:value");
    k2LeftTag      = XMLString::transcode("kactus2:left");
    k2RightTag     = XMLString::transcode("kactus2:right");
    ipxCompGensTag = XMLString::transcode("ipxact:componentGenerators");
    ipxGenExeTag   = XMLString::transcode("ipxact:generatorExe");
    ipxFileTag     = XMLString::transcode("ipxact:file");
  }
public:
  virtual ~IPXACTParser() {
    delete(nameAttr);
    delete(ipxPortTag);
    XMLPlatformUtils::Terminate(); }
private:
  std::map<std::string, std::string> readFileNames; // TODO what is this
  std::string libraryPath; // TODO there might be several lib pathes

  // Attributes.
  XMLCh *nameAttr;
  // IP-XACT common tags.
  XMLCh *ipxPortTag, *ipxNameTag, *ipxVlnvTag, *ipxDirectTag,
        *ipxVectTag, *ipxLeftTag, *ipxRightTag;
  // IP-XACT vendor extensions tags (kactus2).
  XMLCh *k2ParamTag, *k2NameTag, *k2ValueTag, *k2LeftTag, *k2RightTag;
  // For component generators.
  XMLCh *ipxCompGensTag, *ipxGenExeTag;
  // For static compomonents.
  XMLCh *ipxFileTag;
};

} // namespace eda::hls::library
