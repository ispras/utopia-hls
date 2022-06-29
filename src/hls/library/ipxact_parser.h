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

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>

using namespace eda::hls::mapper;
using namespace xercesc;

namespace eda::hls::library {

class IPXACTParser final : public Singleton<IPXACTParser> {
  friend Singleton<IPXACTParser>;

public:
  void initialize();
  void finalize();

  bool hasComponent(const std::string &name, const HWConfig &hwconfig);

  /**
   * @brief Parses an IP-XACT catalog and creates PreMetaElements.
   *
   * PreMetaElement is a structure which hold the following information:
   * path to IP-XACT components, library path, (not)supported HWConfig
   * and estimationfunction.
   *
   * @param libraryPath IP-XACT library path.
   * @param catalogPath IP-XACT catalog path relative to IP-XACT library path.
   * @return Nothing, but creates a map of PreMetaElements.
   */
  void parseCatalog(const std::string &libPath, const std::string &catalogPath);
  /**
   * @brief Parses an IP-XACT component and creates MetaElement.
   *
   * @param name IP-XACT component name.
   * @return MetaElement, which corresponds to the IP-XACT component.
   */
  std::shared_ptr<MetaElement> parseComponent(const std::string &name);

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
/**
 * @brief Parses a tag to get a string value.
 *
 * @param element A DOMElement in which the tag is located.
 * @param tagName Tag name.
 * @return Tag value.
 */
  std::string getStrValueFromTag(const DOMElement *element,
                                 const XMLCh      *tagName);
 /**
  * @brief Parses a tag to get an integer value.
  *
  * @param element A DOMElement in which the tag is located.
  * @param tagName Tag name.
  * @return Tag value.
  */
  int         getIntValueFromTag(const DOMElement *element,
                                 const XMLCh      *tagName);
 /**
  * @brief Parses a tag to get a string value from the given attribute.
  *
  * @param element A DOMElement in which the tag is located.
  * @param tagName Tag name.
  * @return Tag value.
  */
  std::string getStrAttributeFromTag(const DOMElement *element,
                                     const XMLCh      *tagName,
                                     const XMLCh      *attributeName);
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

    XMLPlatformUtils::Terminate();
  }
private:
  std::map<std::string, std::string> readFileNames; // TODO what is this
  /*Replace libraryPath and readFileNames and make a map.*/
  /*std::map<std::string, std::map<std::string, std::string>> compFileNames;*/
  std::string libraryPath; // TODO there might be several lib paths

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
