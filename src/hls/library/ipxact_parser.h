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

  //bool hasComponent(const std::string &name, const std::string &libraryName);

  void parseCatalog(const std::string &libPath, const std::string &catalogPath);
  std::shared_ptr<MetaElement> parseComponent(const std::string &name/*,
                                              const std::string &libraryName*/);

private:
  IPXACTParser() {
    XMLPlatformUtils::Initialize();
    ipxPortTag     = XMLString::transcode("ipxact:port");
    ipxNameTag     = XMLString::transcode("ipxact:name");
    ipxVlnvTag     = XMLString::transcode("ipxact:vlnv");
    ipxDirectTag   = XMLString::transcode("ipxact:direction");
    ipxVectTag     = XMLString::transcode("ipxact:vector");
    ipxLeftTag     = XMLString::transcode("ipxact:left");
    ipxRightTag    = XMLString::transcode("ipxact:right");
    ipxCompGensTag = XMLString::transcode("ipxact:componentGenerators");
    ipxGenExeTag   = XMLString::transcode("ipxact:generatorExe");
    ipxIpxFileTag  = XMLString::transcode("ipxact:ipxactFile");
    ipxFileTag     = XMLString::transcode("ipxact:file");

    k2ParamTag     = XMLString::transcode("kactus2:parameter");
    k2NameTag      = XMLString::transcode("kactus2:name");
    k2ValueTag     = XMLString::transcode("kactus2:value");
    k2LeftTag      = XMLString::transcode("kactus2:left");
    k2RightTag     = XMLString::transcode("kactus2:right");

    nameAttr       = XMLString::transcode("name");
  }
public:
  virtual ~IPXACTParser() {
    delete(ipxPortTag);
    delete(ipxNameTag);
    delete(ipxVlnvTag);
    delete(ipxDirectTag);
    delete(ipxVectTag);
    delete(ipxLeftTag);
    delete(ipxRightTag);
    delete(ipxCompGensTag);
    delete(ipxGenExeTag);
    delete(ipxIpxFileTag);
    delete(ipxFileTag);

    delete(k2ParamTag);
    delete(k2NameTag);
    delete(k2ValueTag);
    delete(k2LeftTag);
    delete(k2RightTag);

    delete(nameAttr);

    XMLPlatformUtils::Terminate(); }
private:
  std::map<std::string, std::string> readFileNames; // TODO what is this
  /*Replace libraryPath and readFileNames and make a map.*/
  /*std::map<std::string, std::map<std::string, std::string>> compFileNames;*/
  std::string libraryPath; // TODO there might be several lib pathes

  // Attributes.
  XMLCh *nameAttr;
  // IP-XACT common tags.
  XMLCh *ipxPortTag, *ipxNameTag, *ipxVlnvTag, *ipxDirectTag,
        *ipxVectTag, *ipxLeftTag, *ipxRightTag;
  // IP-XACT vendor extensions tags (kactus2).
  XMLCh *k2ParamTag, *k2NameTag, *k2ValueTag, *k2LeftTag, *k2RightTag;
  // For catalog.
  XMLCh *ipxIpxFileTag;
  // For component generators.
  XMLCh *ipxCompGensTag, *ipxGenExeTag;
  // For static compomonents.
  XMLCh *ipxFileTag;

};

} // namespace eda::hls::library
