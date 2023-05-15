//===----------------------------------------------------------------------===//
//
// Part of the Utopia EDA Project, under the Apache License v2.0
// SPDX-License-Identifier: Apache-2.0
// Copyright 2022-2023 ISP RAS (http://www.ispras.ru)
//
//===----------------------------------------------------------------------===//

#pragma once

#include "hls/library/library.h"
#include "util/singleton.h"

#include <iostream>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>

using DOMElement = xercesc::DOMElement;
using DOMException = xercesc::DOMException;
using DOMDocument = xercesc::DOMDocument;
using DOMParser = xercesc::ErrorHandler;
using ErrorHandler = xercesc::ErrorHandler;
using HandlerBase = xercesc::HandlerBase;
using SAXException = xercesc::SAXException;
using XercesDOMParser = xercesc::XercesDOMParser;
using XMLException = xercesc::XMLException;
using XMLPlatformUtils = xercesc::XMLPlatformUtils;
using XMLString = xercesc::XMLString;

namespace eda::hls::library {

/**
 * @brief Parser for IP-XACT specification files.
 * @author <a href="mailto:grigorovia@ispras.ru">Ivan Grigorov</a>
 */
class IPXACTParser final : public Singleton<IPXACTParser> {
  friend Singleton<IPXACTParser>;
private:
  IPXACTParser() {
    // Initializes XMLPlatformUtils.
    XMLPlatformUtils::Initialize();

    // Creates Xercecs DOM Parser.
    parser = new XercesDOMParser();
    parser->setValidationScheme(XercesDOMParser::Val_Always);
    parser->setDoNamespaces(true);

    errorHandler = (ErrorHandler*) new HandlerBase();
    parser->setErrorHandler(errorHandler);

    // Tags and attributes initialization.
    ipxPortTag = XMLString::transcode("ipxact:port");
    ipxQualifierTag = XMLString::transcode("ipxact:qualifier");
    ipxIsDataTag = XMLString::transcode("ipxact:isData");
    ipxIsClockTag = XMLString::transcode("ipxact:isClock");
    ipxIsResetTag = XMLString::transcode("ipxact:isReset");
    ipxNameTag = XMLString::transcode("ipxact:name");
    ipxModuleNameTag = XMLString::transcode("ipxact:moduleName");
    ipxLibTag = XMLString::transcode("ipxact:library");
    ipxVlnvTag = XMLString::transcode("ipxact:vlnv");
    ipxDirectTag = XMLString::transcode("ipxact:direction");
    ipxVectTag = XMLString::transcode("ipxact:vector");
    ipxLeftTag = XMLString::transcode("ipxact:left");
    ipxRightTag = XMLString::transcode("ipxact:right");
    ipxCompGensTag = XMLString::transcode("ipxact:componentGenerators");
    ipxGenExeTag = XMLString::transcode("ipxact:generatorExe");
    ipxIpxFileTag = XMLString::transcode("ipxact:ipxactFile");
    ipxFileTag = XMLString::transcode("ipxact:file");
    ipxCompTag = XMLString::transcode("ipxact:component");

    k2ParamTag = XMLString::transcode("kactus2:parameter");
    k2NameTag = XMLString::transcode("kactus2:name");
    k2ValueTag = XMLString::transcode("kactus2:value");
    k2LeftTag = XMLString::transcode("kactus2:left");
    k2RightTag = XMLString::transcode("kactus2:right");

    nameAttr = XMLString::transcode("name");
  }

  bool tryToParseXML(const char* fileName);

  std::string getStrValueFromTag(const DOMElement *element,
                                 const XMLCh *tagName);

  int         getIntValueFromTag(const DOMElement *element,
                                 const XMLCh *tagName);

  bool       getBoolValueFromTag(const DOMElement *element,
                                 const XMLCh *tagName);

  std::string getStrAttrFromTag(const DOMElement *element,
                                const XMLCh *tagName,
                                const XMLCh *attributeName);

  /// Parser.
  XercesDOMParser* parser;
  /// Error handler for parser.
  ErrorHandler* errorHandler;
  /// Attributes.
  XMLCh *nameAttr;
  /// IP-XACT common tags.
  XMLCh *ipxPortTag, *ipxQualifierTag, *ipxIsDataTag, *ipxIsClockTag, 
        *ipxIsResetTag, *ipxNameTag, *ipxVlnvTag, *ipxDirectTag, *ipxVectTag, 
        *ipxLeftTag, *ipxRightTag, *ipxLibTag;
  /// IP-XACT vendor extensions tags (kactus2).
  XMLCh *k2ParamTag, *k2NameTag, *k2ValueTag, *k2LeftTag, *k2RightTag;
  /// For catalog.
  XMLCh *ipxIpxFileTag;
  /// For component generators.
  XMLCh *ipxCompGensTag, *ipxGenExeTag;
  /// For static compomonents.
  XMLCh *ipxFileTag, *ipxCompTag, *ipxModuleNameTag;

public:
  virtual ~IPXACTParser() {
    delete(ipxPortTag);
    delete(ipxQualifierTag);
    delete(ipxIsDataTag);
    delete(ipxIsClockTag);
    delete(ipxIsResetTag);
    delete(ipxLeftTag);
    delete(ipxRightTag);

    delete(ipxNameTag);
    delete(ipxModuleNameTag);
    delete(ipxVlnvTag);
    delete(ipxDirectTag);
    delete(ipxVectTag);
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

};

} // namespace eda::hls::library