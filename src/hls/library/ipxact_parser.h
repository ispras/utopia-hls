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
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>

using namespace eda::hls::mapper;
using namespace xercesc;

namespace eda::hls::library {

class IPXACTParser final : public Singleton<IPXACTParser> {
  friend Singleton<IPXACTParser>;

public:
  /**
   * @brief Constructs meta-elements from an IP-XACT delivery.
   *
   * @param libraryPath IP-XACT library path.
   * @param catalogPath IP-XACT catalog path relative to IP-XACT library path.
   * 
   * @returns Constructed meta-elements.
   */
  std::vector<std::shared_ptr<MetaElement>> getDelivery(
      const std::string &libraryPath, const std::string &catalogPath);

private:
  IPXACTParser() {
    // Initializes XMLPlatformUtils
    XMLPlatformUtils::Initialize();

    // Create Xercecs DOM Parser
    parser = new XercesDOMParser();
    parser->setValidationScheme(XercesDOMParser::Val_Always);
    parser->setDoNamespaces(true);
    errorHandler = (ErrorHandler*) new HandlerBase();
    parser->setErrorHandler(errorHandler);

    // Tags and attributes initialization
    ipxPortTag     = XMLString::transcode("ipxact:port");
    ipxNameTag     = XMLString::transcode("ipxact:name");
    ipxLibTag      = XMLString::transcode("ipxact:library");
    ipxVlnvTag     = XMLString::transcode("ipxact:vlnv");
    ipxDirectTag   = XMLString::transcode("ipxact:direction");
    ipxVectTag     = XMLString::transcode("ipxact:vector");
    ipxLeftTag     = XMLString::transcode("ipxact:left");
    ipxRightTag    = XMLString::transcode("ipxact:right");
    ipxCompGensTag = XMLString::transcode("ipxact:componentGenerators");
    ipxGenExeTag   = XMLString::transcode("ipxact:generatorExe");
    ipxIpxFileTag  = XMLString::transcode("ipxact:ipxactFile");
    ipxFileTag     = XMLString::transcode("ipxact:file");
    ipxCompTag     = XMLString::transcode("ipxact:component");

    k2ParamTag     = XMLString::transcode("kactus2:parameter");
    k2NameTag      = XMLString::transcode("kactus2:name");
    k2ValueTag     = XMLString::transcode("kactus2:value");
    k2LeftTag      = XMLString::transcode("kactus2:left");
    k2RightTag     = XMLString::transcode("kactus2:right");

    nameAttr       = XMLString::transcode("name");
  }
  /**
   * @brief Tries to parse an XML file and to construct a DOM document.
   *
   * @param fileName XML file to parse.
   * 
   * @returns true if XML file is correct, false otherwise.
   */
  bool tryToParseXML(const char* fileName);

  /**
   * @brief Parses a tag to get a string value.
   *
   * @param element DOMElement in which tag is located.
   * @param tagName Tag name.
   * 
   * @returns Tag value.
   */
  std::string getStrValueFromTag(const DOMElement *element,
                                 const XMLCh      *tagName);
                                 
  /**
   * @brief Parses a tag to get an integer value.
   *
   * @param element DOMElement in which tag is located.
   * @param tagName Tag name.
   * 
   * @returns Tag value.
   */
  int         getIntValueFromTag(const DOMElement *element,
                                 const XMLCh      *tagName);

  /**
   * @brief Parses a tag to get a string value from the given attribute.
   *
   * @param element DOMElement in which tag is located.
   * @param tagName Tag name.
   * @param attributeName Attribute name.
   * 
   * @returns Attribute value.
   */
  std::string getStrAttributeFromTag(const DOMElement *element,
                                     const XMLCh      *tagName,
                                     const XMLCh      *attributeName);

  /**
   * @brief Parses an IP-XACT catalog to get the components' filenames.
   *
   * @param libraryPath IP-XACT library path.
   * @param catalogPath IP-XACT catalog path relative to IP-XACT library.
   * 
   * @returns Components' filenames relative to IP-XACT library.
   */
  std::vector<std::string> parseCatalog(const std::string &libraryPath,
                                        const std::string &catalogPath);
  
  /**
   * @brief Parses IP-XACT components and creates the set of meta-elements.
   *
   * @param libraryPath    IP-XACT library path.
   * @param componentPaths IP-XACT components paths relative to IP-XACT library.
   * 
   * @returns Constructed meta-elements.
   */
  std::vector<std::shared_ptr<MetaElement>> parseComponents(
      const std::string              &libraryPath,
      const std::vector<std::string> &componentPaths);

  /**
   * @brief Parses an IP-XACT component and creates the meta-element.
   *
   * @param libraryPath   IP-XACT library path.
   * @param componentPath IP-XACT component path relative to IP-XACT library.
   * 
   * @returns Constructed meta-element.
   */
  std::shared_ptr<MetaElement> parseComponent(const std::string &libraryPath,
                                              const std::string &componentPath);

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
    delete(ipxCompTag);
    delete(ipxLibTag);

    delete(k2ParamTag);
    delete(k2NameTag);
    delete(k2ValueTag);
    delete(k2LeftTag);
    delete(k2RightTag);

    delete(nameAttr);

    XMLPlatformUtils::Terminate();
  }
private:
  /// Parser
  XercesDOMParser* parser;
  /// Error handler for parser
  ErrorHandler* errorHandler;
  /// Attributes
  XMLCh *nameAttr;
  /// IP-XACT common tags
  XMLCh *ipxPortTag, *ipxNameTag, *ipxVlnvTag, *ipxDirectTag,
        *ipxVectTag, *ipxLeftTag, *ipxRightTag, *ipxLibTag;
  /// IP-XACT vendor extensions tags (kactus2)
  XMLCh *k2ParamTag, *k2NameTag, *k2ValueTag, *k2LeftTag, *k2RightTag;
  /// For catalog
  XMLCh *ipxIpxFileTag;
  /// For component generators
  XMLCh *ipxCompGensTag, *ipxGenExeTag;
  /// For static compomonents
  XMLCh *ipxFileTag, *ipxCompTag;
};

} // namespace eda::hls::library
